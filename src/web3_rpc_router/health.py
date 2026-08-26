from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

from web3_rpc_router.provider import ProviderState

logger = logging.getLogger("web3_rpc_router")

# Consecutive failed health checks a provider is allowed before it is considered down.
_UNHEALTHY_AFTER_FAILURES = 3
# Ceiling on the probe backoff for a provider that stays down, in seconds.
_BACKOFF_MAX = 300.0
# Bound the shift so a long-dead provider cannot build a needlessly large integer
# before ``min`` discards it.
_BACKOFF_MAX_DOUBLINGS = 16


class HealthChecker:
    """Background task that periodically checks provider health via block number."""

    def __init__(
        self,
        providers: Dict[int, List[ProviderState]],
        interval: float,
        max_block_lag: int,
        timeout: float,
        retry_interval: float = 30.0,
    ) -> None:
        self._providers = providers
        self._interval = interval
        self._retry_interval = retry_interval
        self._max_block_lag = max_block_lag
        self._timeout = timeout
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Start the background health check loop."""
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> Optional[asyncio.Task]:
        """Cancel the background loop and return its task so callers can await it.

        The task is returned rather than dropped because a cycle may be mid-probe:
        tearing down the sessions and resolver it is using before it has unwound would
        have it fail against closed objects and write verdicts after shutdown.
        """
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
        return task

    def _has_unhealthy(self) -> bool:
        """Return True if any provider across all chains is unhealthy."""
        return any(
            not p.healthy for providers in self._providers.values() for p in providers
        )

    async def _loop(self) -> None:
        while True:
            sleep = self._retry_interval if self._has_unhealthy() else self._interval
            await asyncio.sleep(sleep)
            try:
                await self.check_all()
            except Exception:
                logger.exception("Health check cycle failed")

    async def check_all(self) -> None:
        """Check every provider that is due, across all chains, in parallel.

        A provider that has failed enough consecutive checks earns a backoff window
        and is skipped until it expires, so a permanently dead endpoint costs one
        probe per ``_BACKOFF_MAX`` instead of one per cycle. Its existing verdict
        stands while it is skipped, and one successful check clears the backoff.

        Backoff is ignored on a chain with no healthy provider left. Such a chain is
        being served in degraded mode, so noticing any provider come back outweighs
        sparing a dead endpoint, and waiting out a capped window there would leave
        selection stuck on ``providers[0]`` even once another provider recovered.
        """
        due = time.time()
        all_providers = []
        for chain_id, providers in self._providers.items():
            chain_is_dark = not any(p.healthy for p in providers)
            all_providers.extend(
                (chain_id, p) for p in providers if chain_is_dark or p.next_check <= due
            )
        if not all_providers:
            return

        cooldowns_at_probe_time = {id(p): p.cooldown_until for _, p in all_providers}
        results = await asyncio.gather(
            *(self._check_one(p) for _, p in all_providers),
            return_exceptions=True,
        )

        # Group results back by chain
        chain_results: Dict[int, List[tuple]] = {}
        for (chain_id, p), result in zip(all_providers, results):
            chain_results.setdefault(chain_id, []).append((p, result))

        now = time.time()

        # Process results per chain
        for chain_id, provider_results in chain_results.items():
            max_block = 0
            for p, result in provider_results:
                # Anything that is not a block number counts as a failure. Testing for
                # Exception would miss a cancelled probe: asyncio.gather returns a
                # CancelledError as a *result*, and it derives from BaseException, so it
                # would be stored as last_block and make every later max() over the
                # chain's blocks raise.
                if not isinstance(result, int):
                    logger.warning(
                        "Health check failed for %s (chain %d): %r",
                        p.config.name,
                        chain_id,
                        result,
                    )
                    p.consecutive_failures += 1
                    backoff = self._backoff_delay(p.consecutive_failures)
                    p.next_check = now + backoff
                    if backoff:
                        logger.debug(
                            "Provider %s (chain %d) backing off %.0fs after %d "
                            "consecutive failures",
                            p.config.name,
                            chain_id,
                            backoff,
                            p.consecutive_failures,
                        )
                else:
                    p.last_block = result
                    p.consecutive_failures = 0
                    p.next_check = 0.0
                    # Clear any request-level cooldown: if the provider is answering
                    # again, let the router consider it fresh. Only if it is the same
                    # cooldown that was in place when this probe started, since
                    # report_failure() can demote the provider while a probe is in
                    # flight and that demotion must survive a stale success.
                    if p.cooldown_until == cooldowns_at_probe_time.get(id(p)):
                        p.cooldown_until = 0.0
                    max_block = max(max_block, result)

            if max_block == 0:
                max_block = max((p.last_block for p, _ in provider_results), default=0)

            for p, _ in provider_results:
                was_healthy = p.healthy
                if p.consecutive_failures >= _UNHEALTHY_AFTER_FAILURES:
                    p.healthy = False
                elif p.last_block == 0:
                    p.healthy = False
                else:
                    p.healthy = (max_block - p.last_block) <= self._max_block_lag
                p.last_check = now

                if was_healthy and not p.healthy:
                    logger.warning(
                        "Provider %s (chain %d) marked UNHEALTHY (block %d, max %d)",
                        p.config.name,
                        chain_id,
                        p.last_block,
                        max_block,
                    )
                elif not was_healthy and p.healthy:
                    logger.info(
                        "Provider %s (chain %d) recovered (block %d)",
                        p.config.name,
                        chain_id,
                        p.last_block,
                    )

    def _backoff_delay(self, failures: int) -> float:
        """Seconds to skip a provider that has failed ``failures`` checks in a row.

        Zero while a provider is still within its allowance, so a brief blip is
        retried on the normal cadence. Past that the wait doubles each failure up to
        ``_BACKOFF_MAX``, which keeps a dead endpoint from being probed every cycle
        while still recovering it within one capped window of coming back.
        """
        if failures < _UNHEALTHY_AFTER_FAILURES:
            return 0.0
        doublings = min(failures - _UNHEALTHY_AFTER_FAILURES, _BACKOFF_MAX_DOUBLINGS)
        return min(_BACKOFF_MAX, self._retry_interval * (2**doublings))

    async def _check_one(self, p: ProviderState) -> int:
        """Query a single provider's block number over its pooled probe client.

        The probe runs entirely on the event loop, so a provider that misses
        ``self._timeout`` is abandoned by ``asyncio.wait_for`` leaving nothing
        behind. That matters because the underlying provider timeout
        (``ProviderConfig.request_timeout``) is far longer than the health
        budget: a probe offloaded to a worker thread cannot be cancelled, so it
        would keep occupying a slot in asyncio's default executor for the full
        request timeout. aiohttp resolves DNS and decompresses response bodies
        in that same pool, so a handful of hanging providers there would starve
        both the remaining providers' probes and real RPC traffic.
        """
        return await asyncio.wait_for(
            p.probe_w3.eth.block_number,
            timeout=self._timeout,
        )

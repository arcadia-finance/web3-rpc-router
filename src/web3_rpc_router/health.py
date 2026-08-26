from __future__ import annotations

import asyncio
import json
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

    def _due_providers(self, now: float) -> List[tuple]:
        """``(chain_id, provider)`` pairs to probe on this cycle.

        A provider inside an open backoff window is skipped, unless its chain has no
        healthy provider left. Such a chain is being served in degraded mode via
        ``providers[0]``, so noticing any recovery outweighs sparing a dead endpoint,
        and waiting out a capped window there would leave selection pinned even after
        another provider started answering.
        """
        due: List[tuple] = []
        for chain_id, providers in self._providers.items():
            chain_is_dark = not any(p.healthy for p in providers)
            due.extend(
                (chain_id, p) for p in providers if chain_is_dark or p.next_check <= now
            )
        return due

    def _has_unhealthy(self) -> bool:
        """Return True while an unhealthy provider is due to be probed again.

        This picks the loop's cadence. The shorter ``retry_interval`` exists to bring a
        provider back quickly, so a provider whose backoff window is still open must not
        hold the whole fleet at it: otherwise a single permanently dead endpoint pins
        every chain to ``retry_interval`` for the life of the process. Reading the same
        due list ``check_all`` will act on keeps the cadence matched to the work, so a
        chain with nothing healthy still gets the fast cadence it needs.
        """
        return any(not p.healthy for _, p in self._due_providers(time.time()))

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
        all_providers = self._due_providers(time.time())
        if not all_providers:
            return

        cooldowns_at_probe_time = {id(p): p.cooldown_until for _, p in all_providers}
        results = await asyncio.gather(
            *(self._check_one(p) for _, p in all_providers),
            return_exceptions=True,
        )

        # A cycle in which not one probed provider answered, while at least one
        # of them has answered before, is overwhelmingly a local measurement gap
        # — the event loop stalled past every probe's deadline (CPU-throttled
        # quiet instance, heavy on-loop work) or the instance's egress dropped —
        # not twenty independent provider outages. Recording it would strike and
        # eventually condemn every provider for evidence that says nothing about
        # any of them, so the cycle is discarded: no failure counts, no backoff,
        # no verdicts. A cycle where anything answered proves the loop and the
        # network were alive, so its failures are real evidence and are counted
        # below. A cold start (nothing has ever answered) is not shielded:
        # providers that never answered have no known-good state to preserve.
        if not any(isinstance(r, int) for r in results) and any(
            p.last_block > 0 for _, p in all_providers
        ):
            logger.warning(
                "Health check cycle got no answer from any of %d providers; "
                "treating it as a measurement gap, all verdicts stand",
                len(all_providers),
            )
            return

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

            # max_block is the best block any provider proved this cycle. Providers that
            # did not answer are deliberately excluded: comparing their stale last_block
            # against a peer's fresh one would mark a provider dead for missing a single
            # probe, which is a measurement gap rather than evidence about the provider.
            for p, result in provider_results:
                was_healthy = p.healthy
                if isinstance(result, int):
                    # Answered, so judge it on this cycle's evidence.
                    p.healthy = (max_block - p.last_block) <= self._max_block_lag
                elif p.consecutive_failures >= _UNHEALTHY_AFTER_FAILURES:
                    # Only sustained failure is evidence of death.
                    p.healthy = False
                elif p.last_block == 0:
                    # Never answered, so there is no known-good state to preserve and
                    # nothing has earned it the benefit of the doubt.
                    p.healthy = False
                # Otherwise the previous verdict stands: a provider that has proved it
                # works does not lose its health for one missed probe, because a gap in
                # measurement says nothing about the provider.
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
        """Query a single provider's block number over its pooled session.

        The probe runs entirely on the event loop, so a provider that misses
        ``self._timeout`` is abandoned by ``asyncio.wait_for`` leaving nothing
        behind. That matters because the underlying provider timeout
        (``ProviderConfig.request_timeout``) is far longer than the health
        budget: a probe offloaded to a worker thread cannot be cancelled, so it
        would keep occupying a slot in asyncio's default executor for the full
        request timeout. aiohttp resolves DNS and decompresses response bodies
        in that same pool, so a handful of hanging providers there would starve
        both the remaining providers' probes and real RPC traffic.

        The request is a plain aiohttp POST rather than a web3 call, and that is
        load-bearing: web3 serialises every async request through one
        process-wide ``threading.Lock`` acquired on an executor thread, and
        cancelling a task while its acquire is in flight abandons the acquire,
        leaving the lock held forever and every later web3 request in the
        process blocked behind it. Probes are cancelled by design whenever a
        provider (or a CPU-throttled instance) is slow, so they must not pass
        through that lock at all. A cancelled aiohttp request only costs its own
        pooled connection. See ``ProviderState.probe_session``.
        """
        return await asyncio.wait_for(self._probe(p), timeout=self._timeout)

    @staticmethod
    async def _probe(p: ProviderState) -> int:
        session = p.probe_session
        if session is None:
            raise RuntimeError(
                f"provider {p.config.name} has no probe session; "
                "RPCRouter.start() assigns it"
            )
        payload = {"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}
        async with session.post(p.config.url, json=payload) as resp:
            body = await resp.read()
        # A JSON-RPC error response has no "result"; a malformed body does not
        # parse. Both raise here and are counted as a failed probe by check_all.
        return int(json.loads(body)["result"], 16)

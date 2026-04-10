from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

from web3_rpc_router.provider import ProviderState

logger = logging.getLogger("web3_rpc_router")


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

    def stop(self) -> None:
        """Cancel the background health check loop."""
        if self._task:
            self._task.cancel()
            self._task = None

    def _has_unhealthy(self) -> bool:
        """Return True if any provider across all chains is unhealthy."""
        return any(
            not p.healthy
            for providers in self._providers.values()
            for p in providers
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
        """Check all providers across all chains in parallel."""
        # Check all providers across all chains concurrently
        all_providers = [
            (chain_id, p)
            for chain_id, providers in self._providers.items()
            for p in providers
        ]
        results = await asyncio.gather(
            *(self._check_one(p) for _, p in all_providers),
            return_exceptions=True,
        )

        # Group results back by chain
        chain_results: Dict[int, List[tuple]] = {}
        for (chain_id, p), result in zip(all_providers, results):
            chain_results.setdefault(chain_id, []).append((p, result))

        # Process results per chain
        for chain_id, provider_results in chain_results.items():
            max_block = 0
            for p, result in provider_results:
                if isinstance(result, Exception):
                    logger.warning(
                        "Health check failed for %s (chain %d): %r",
                        p.config.name, chain_id, result,
                    )
                    p._consecutive_failures = getattr(p, "_consecutive_failures", 0) + 1
                else:
                    p.last_block = result
                    p._consecutive_failures = 0
                    max_block = max(max_block, result)

            if max_block == 0:
                max_block = max((p.last_block for p, _ in provider_results), default=0)

            now = time.time()
            for p, _ in provider_results:
                was_healthy = p.healthy
                failures = getattr(p, "_consecutive_failures", 0)
                if failures >= 3:
                    p.healthy = False
                elif p.last_block == 0:
                    p.healthy = False
                else:
                    p.healthy = (max_block - p.last_block) <= self._max_block_lag
                p.last_check = now

                if was_healthy and not p.healthy:
                    logger.warning(
                        "Provider %s (chain %d) marked UNHEALTHY " "(block %d, max %d)",
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

    async def _check_one(self, p: ProviderState) -> int:
        """Query a single provider's block number using a sync Web3 instance
        in a thread, avoiding aiohttp session issues."""
        return await asyncio.wait_for(
            asyncio.to_thread(lambda: p.health_w3.eth.block_number),
            timeout=self._timeout,
        )

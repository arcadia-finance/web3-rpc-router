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
    ) -> None:
        self._providers = providers
        self._interval = interval
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

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.check_all()
            except Exception:
                logger.exception("Health check cycle failed")

    async def check_all(self) -> None:
        """Check all providers across all chains."""
        for chain_id, providers in self._providers.items():
            results = await asyncio.gather(
                *(self._check_one(p) for p in providers),
                return_exceptions=True,
            )

            # Find highest block number
            max_block = 0
            for p, result in zip(providers, results):
                if isinstance(result, Exception):
                    p.last_block = 0
                else:
                    p.last_block = result
                    max_block = max(max_block, result)

            # Update health status
            now = time.time()
            for p in providers:
                was_healthy = p.healthy
                if p.last_block == 0:
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
        """Query a single provider's block number using async web3."""
        return await asyncio.wait_for(
            p.async_w3.eth.get_block_number(),
            timeout=self._timeout,
        )

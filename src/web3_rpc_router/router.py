from __future__ import annotations

import logging
from typing import Dict, List, Optional

from web3 import AsyncWeb3, Web3

from web3_rpc_router.health import HealthChecker
from web3_rpc_router.provider import ProviderConfig, ProviderState

logger = logging.getLogger("web3_rpc_router")


class RPCRouter:
    """Multi-provider RPC router with health-based selection.

    Usage::

        router = RPCRouter(check_interval=900, max_block_lag=1)
        router.add_provider(8453, ProviderConfig(name="local", url="...", priority=1))
        router.add_provider(8453, ProviderConfig(name="alchemy", url="...", priority=2))
        await router.start()

        w3 = router.get_web3(8453)              # sync Web3
        async_w3 = router.get_async_web3(8453)   # async AsyncWeb3

        await router.stop()
    """

    def __init__(
        self,
        check_interval: float = 900.0,
        max_block_lag: int = 1,
        health_check_timeout: float = 5.0,
    ) -> None:
        self._check_interval = check_interval
        self._max_block_lag = max_block_lag
        self._health_check_timeout = health_check_timeout
        self._providers: Dict[int, List[ProviderState]] = {}
        self._health_checker: Optional[HealthChecker] = None
        self._started = False

    def add_provider(self, chain_id: int, config: ProviderConfig) -> None:
        """Register a provider for a chain. Call before start()."""
        if chain_id not in self._providers:
            self._providers[chain_id] = []
        state = ProviderState(config=config)
        self._providers[chain_id].append(state)
        self._providers[chain_id].sort(key=lambda s: s.config.priority)

    async def start(self) -> None:
        """Run initial health check and start background checker."""
        if self._started:
            return
        self._health_checker = HealthChecker(
            providers=self._providers,
            interval=self._check_interval,
            max_block_lag=self._max_block_lag,
            timeout=self._health_check_timeout,
        )
        await self._health_checker.check_all()
        self._health_checker.start()
        self._started = True

    async def stop(self) -> None:
        """Stop the background health checker."""
        if self._health_checker:
            self._health_checker.stop()
        self._started = False

    @property
    def chain_ids(self) -> List[int]:
        """Return all configured chain IDs."""
        return list(self._providers.keys())

    def _select_provider(self, chain_id: int) -> ProviderState:
        """Select the best healthy provider for the given chain."""
        providers = self._providers.get(chain_id)
        if not providers:
            raise ValueError(f"No providers configured for chain {chain_id}")

        max_block = max(p.last_block for p in providers)

        for p in providers:
            if p.healthy:
                behind = max_block - p.last_block
                logger.info(
                    "Chain %d → selected provider: %s (priority %d, block %d, %d behind head)",
                    chain_id,
                    p.config.name,
                    p.config.priority,
                    p.last_block,
                    behind,
                )
                return p

        logger.warning(
            "All providers unhealthy for chain %d, using %s in degraded mode",
            chain_id,
            providers[0].config.name,
        )
        return providers[0]

    def get_web3(self, chain_id: int) -> Web3:
        """Return the best sync Web3 instance for the given chain.

        Returns the highest-priority healthy provider.
        Falls back to highest-priority provider if all are unhealthy.

        Raises:
            ValueError: If no providers are configured for the chain.
        """
        return self._select_provider(chain_id).w3

    def get_async_web3(self, chain_id: int) -> AsyncWeb3:
        """Return the best async AsyncWeb3 instance for the given chain.

        Same selection logic as get_web3, but returns an AsyncWeb3 instance.

        Raises:
            ValueError: If no providers are configured for the chain.
        """
        return self._select_provider(chain_id).async_w3

    def get_provider_status(self, chain_id: int) -> List[dict]:
        """Return status of all providers for a chain (for monitoring)."""
        return [
            {
                "name": p.config.name,
                "priority": p.config.priority,
                "healthy": p.healthy,
                "last_block": p.last_block,
                "last_check": p.last_check,
            }
            for p in self._providers.get(chain_id, [])
        ]

    def status(self) -> Dict[int, List[dict]]:
        """Return a summary of all providers across all chains.

        Returns a dict keyed by chain_id, each containing a list of provider
        status dicts with a ``behind`` field showing blocks behind the chain head.
        """
        result: Dict[int, List[dict]] = {}
        for chain_id, providers in self._providers.items():
            max_block = max((p.last_block for p in providers), default=0)
            result[chain_id] = [
                {
                    "name": p.config.name,
                    "priority": p.config.priority,
                    "healthy": p.healthy,
                    "last_block": p.last_block,
                    "behind": max_block - p.last_block,
                    "last_check": p.last_check,
                }
                for p in providers
            ]
        return result

    def log_status(self) -> None:
        """Log a human-readable summary of all providers."""
        for chain_id, providers in self.status().items():
            lines = []
            for p in providers:
                health = "OK" if p["healthy"] else "DOWN"
                lines.append(
                    f"  {p['name']} (pri={p['priority']}): {health}, "
                    f"block={p['last_block']}, behind={p['behind']}"
                )
            logger.info("Chain %d providers:\n%s", chain_id, "\n".join(lines))

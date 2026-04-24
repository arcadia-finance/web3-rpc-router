from __future__ import annotations

import logging
import time
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
        retry_interval: float = 30.0,
    ) -> None:
        self._check_interval = check_interval
        self._max_block_lag = max_block_lag
        self._health_check_timeout = health_check_timeout
        self._retry_interval = retry_interval
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
            retry_interval=self._retry_interval,
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
        """Select the best provider for the given chain.

        Preference order:
          1. Healthy and not in request-level cooldown.
          2. Healthy but in cooldown (all fresh providers unavailable).
          3. First provider in priority order (degraded mode).
        """
        providers = self._providers.get(chain_id)
        if not providers:
            raise ValueError(f"No providers configured for chain {chain_id}")

        max_block = max(p.last_block for p in providers)
        now = time.time()

        for p in providers:
            if p.healthy and p.cooldown_until <= now:
                behind = max_block - p.last_block
                logger.debug(
                    "Chain %d → selected provider: %s (priority %d, block %d, %d behind head)",
                    chain_id,
                    p.config.name,
                    p.config.priority,
                    p.last_block,
                    behind,
                )
                return p

        for p in providers:
            if p.healthy:
                logger.warning(
                    "Chain %d → all fresh providers cooling down, falling back to %s "
                    "(cooldown expires in %.1fs)",
                    chain_id,
                    p.config.name,
                    max(0.0, p.cooldown_until - now),
                )
                return p

        logger.warning(
            "All providers unhealthy for chain %d, using %s in degraded mode",
            chain_id,
            providers[0].config.name,
        )
        return providers[0]

    def report_failure(self, chain_id: int, cooldown: float = 60.0) -> None:
        """Demote the currently-selected provider for ``chain_id``.

        Called by consumers when a real request (not the background health
        check) fails on whichever provider was handed out most recently —
        timeouts, connection errors, etc. The provider is skipped by
        ``_select_provider`` for ``cooldown`` seconds, giving the retry loop a
        chance to rotate to the next-priority provider. The cooldown is cleared
        automatically by the next successful background health check.

        Safe to call when no providers are configured for the chain (no-op).
        The selection used to identify "currently-selected" is the same logic
        ``get_web3`` / ``get_async_web3`` use, so the demoted provider is the
        one most likely responsible for the failure.
        """
        if not self._providers.get(chain_id):
            return
        try:
            state = self._select_provider(chain_id)
        except ValueError:
            return
        state.cooldown_until = time.time() + cooldown
        logger.warning(
            "Provider %s (chain %d) demoted for %.0fs after request-level failure",
            state.config.name,
            chain_id,
            cooldown,
        )

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
        now = time.time()
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
                    "cooldown_remaining": max(0.0, p.cooldown_until - now),
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
                cooldown = (
                    f", cooldown={p['cooldown_remaining']:.0f}s"
                    if p["cooldown_remaining"] > 0
                    else ""
                )
                lines.append(
                    f"  {p['name']} (pri={p['priority']}): {health}, "
                    f"block={p['last_block']}, behind={p['behind']}{cooldown}"
                )
            logger.info("Chain %d providers:\n%s", chain_id, "\n".join(lines))

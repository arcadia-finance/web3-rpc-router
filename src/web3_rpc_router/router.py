from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

import aiohttp
from aiohttp.resolver import aiodns_default
from web3 import AsyncWeb3, Web3

from web3_rpc_router.health import HealthChecker
from web3_rpc_router.provider import ProviderConfig, ProviderState, RoutingProvider

logger = logging.getLogger("web3_rpc_router")

# Max simultaneous keep-alive connections per provider session. See
# `_init_keepalive_sessions` for why we override web3 7's default connector.
_DEFAULT_CONNECTION_LIMIT = 100


def _build_resolver(use_async_dns: bool) -> "aiohttp.abc.AbstractResolver":
    """Return the DNS resolver for every pooled session.

    Defaults to ``ThreadedResolver`` because it is the only one whose behaviour under
    concurrency is known here. Measured against a shared ``AsyncResolver``, which funnels
    every lookup in the process through one c-ares channel, p50 per lookup was 24ms / 34ms
    / 77ms at 5 / 20 / 50 concurrent, while ``ThreadedResolver`` held 23ms / 4.7ms / 9.7ms.
    Since a lookup is what a new connection waits on, that degradation lands on exactly the
    requests that cannot reuse a warm connection.

    ``use_async_dns`` has to be asked for explicitly. Selecting c-ares merely because
    ``aiodns`` became importable makes adding a dependency change production DNS with no
    diff to review.
    """
    if not use_async_dns:
        return aiohttp.ThreadedResolver()
    if not aiodns_default:
        logger.warning(
            "use_async_dns was requested but there is no usable aiodns; falling back to "
            "ThreadedResolver"
        )
        return aiohttp.ThreadedResolver()
    return aiohttp.AsyncResolver()


def _evict_cached_session(w3: AsyncWeb3) -> None:
    """Drop any session web3 has already cached for this provider.

    ``cache_async_session`` honours the session it is given only on a cache miss. If an
    entry is already there and closed, as it is on a second ``start()`` after a
    ``stop()``, web3 discards the session passed in and caches one of its own built with
    ``force_close=True`` instead. That silently undoes both the connection pooling this
    router seeds and the resolver it chose, so clear the entry first.

    Best effort: the cache is web3-internal and its shape is not part of web3's API.
    """
    try:
        w3.provider._request_session_manager.session_cache.clear()
    except Exception:  # pragma: no cover - depends on web3 internals
        logger.debug("Could not clear web3's session cache; keep-alive may not apply")


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
        connection_limit: int = _DEFAULT_CONNECTION_LIMIT,
        use_async_dns: bool = False,
    ) -> None:
        self._check_interval = check_interval
        self._max_block_lag = max_block_lag
        self._health_check_timeout = health_check_timeout
        self._retry_interval = retry_interval
        self._connection_limit = connection_limit
        self._use_async_dns = use_async_dns
        self._providers: Dict[int, List[ProviderState]] = {}
        self._sessions: List[aiohttp.ClientSession] = []
        self._resolver: Optional["aiohttp.abc.AbstractResolver"] = None
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
        await self._init_keepalive_sessions()
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

    async def _init_keepalive_sessions(self) -> None:
        """Give each provider's async client a keep-alive (connection-pooling) session.

        web3 7's ``AsyncHTTPProvider`` lazily caches an ``aiohttp`` session built with
        ``TCPConnector(force_close=True)``, which disables HTTP keep-alive — every RPC
        call then pays a fresh TCP+TLS handshake. Under the concurrency this router is
        built for (many simultaneous ``eth_call``/multicall requests sharing one
        provider) that serializes into multi-second latencies and request timeouts,
        cascading every provider into cooldown.

        Pre-seeding the provider's session cache with a pooled connector restores
        connection reuse: measured ~30x faster at 50 concurrent calls.
        """
        # One resolver shared by every session. Passing a resolver makes the connector a
        # borrower rather than an owner (it only closes a resolver it built itself), so the
        # router closes this one in stop(); a per-session resolver would leak a registration
        # in aiohttp's shared DNS resolver manager on every start/stop cycle.
        self._resolver = _build_resolver(self._use_async_dns)
        for providers in self._providers.values():
            for p in providers:
                session = aiohttp.ClientSession(
                    raise_for_status=True,
                    connector=aiohttp.TCPConnector(
                        limit=self._connection_limit,
                        resolver=self._resolver,
                    ),
                )
                # Tracked before it is handed over, so a failure below still closes it.
                self._sessions.append(session)
                for w3 in (p.async_w3, p.probe_w3):
                    _evict_cached_session(w3)
                    await w3.provider.cache_async_session(session)

    async def stop(self) -> None:
        """Stop the background health checker, then close the sessions and resolver."""
        if self._health_checker:
            task = self._health_checker.stop()
            if task is not None:
                # Let the cancelled cycle unwind before its sessions go away.
                await asyncio.gather(task, return_exceptions=True)
        for session in self._sessions:
            if not session.closed:
                await session.close()
        self._sessions = []
        # After the sessions, since they resolve through it.
        if self._resolver is not None:
            await self._resolver.close()
            self._resolver = None
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

    def web3(self, chain_id: int) -> Web3:
        """Return a Web3 backed by a RoutingProvider for the chain.

        Unlike get_web3 (which returns a fixed best-provider Web3 captured at call time), every
        request on this Web3 re-selects the current-best provider and fails over on transport
        errors. Use it as a drop-in `Web3` wherever the consumer reads/writes on-chain so the
        call sites stay unchanged but follow the router.
        """
        return Web3(RoutingProvider(self, chain_id))

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
                    # Probes are sparse for a provider that stays down, so last_check
                    # alone cannot distinguish "waiting out a backoff window" from
                    # "the health checker has stopped running".
                    "consecutive_failures": p.consecutive_failures,
                    "backoff_remaining": max(0.0, p.next_check - now),
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
                backoff = (
                    f", backoff={p['backoff_remaining']:.0f}s"
                    f" after {p['consecutive_failures']} failures"
                    if p["backoff_remaining"] > 0
                    else ""
                )
                lines.append(
                    f"  {p['name']} (pri={p['priority']}): {health}, "
                    f"block={p['last_block']}, behind={p['behind']}{cooldown}{backoff}"
                )
            logger.info("Chain %d providers:\n%s", chain_id, "\n".join(lines))

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import requests
from web3 import AsyncWeb3, Web3
from web3.providers.base import JSONBaseProvider
from web3.types import RPCEndpoint, RPCResponse

if TYPE_CHECKING:
    from web3_rpc_router.router import RPCRouter

# Transport-level failures that mean "this provider is bad, try the next one" (as opposed to an
# RPC-level {"error": ...} response, which is deterministic and returned as-is).
_FAILOVER_EXC = (OSError, TimeoutError, requests.exceptions.RequestException)


@dataclass
class ProviderConfig:
    """Configuration for a single RPC provider."""

    name: str
    url: str
    priority: int  # Lower = higher priority (1 is best)
    request_timeout: int = 15


@dataclass
class ProviderState:
    """Runtime state for a provider (internal)."""

    config: ProviderConfig
    w3: Web3 = field(init=False, repr=False)
    async_w3: AsyncWeb3 = field(init=False, repr=False)
    probe_w3: AsyncWeb3 = field(init=False, repr=False)
    healthy: bool = True
    last_block: int = 0
    last_check: float = 0.0
    # Epoch seconds until which this provider is demoted due to a real-request
    # failure reported via RPCRouter.report_failure(). Cleared by a successful
    # background health check.
    cooldown_until: float = 0.0
    # Consecutive failed health checks. Drives both the unhealthy verdict and the
    # probe backoff. Reset to 0 by a successful check.
    consecutive_failures: int = 0
    # Epoch seconds before which the background checker skips probing this provider,
    # so a provider that stays down is polled progressively less often.
    next_check: float = 0.0

    def __post_init__(self) -> None:
        timeout = self.config.request_timeout
        self.w3 = Web3(
            Web3.HTTPProvider(
                self.config.url,
                request_kwargs={"timeout": timeout},
            )
        )
        self.async_w3 = AsyncWeb3(
            AsyncWeb3.AsyncHTTPProvider(
                self.config.url,
                request_kwargs={"timeout": timeout},
            )
        )
        # Health probes only, so they stay one request each. web3 retries
        # eth_blockNumber up to five times with its own backoff, which against a
        # refusing or rate-limited provider would spend five requests and most of the
        # probe's timeout budget to reach a conclusion the first failure already gave.
        # Real traffic keeps web3's retries: it goes through ``async_w3``.
        # ``RPCRouter._init_keepalive_sessions`` points this at the same pooled session,
        # so the extra provider costs no extra connections.
        self.probe_w3 = AsyncWeb3(
            AsyncWeb3.AsyncHTTPProvider(
                self.config.url,
                request_kwargs={"timeout": timeout},
                exception_retry_configuration=None,
            )
        )


class RoutingProvider(JSONBaseProvider):
    """A sync web3 provider that sends each request to the RPCRouter's current-best provider for a
    chain, failing over to the next provider on transport errors.

    Lets a single `Web3` (built via `RPCRouter.web3(chain_id)`) transparently follow the router with
    no consumer call-site changes: every `w3.eth.*` request re-selects the healthy provider. An
    RPC-level ``{"error": ...}`` response is returned as-is (deterministic, not retried).
    """

    def __init__(self, router: "RPCRouter", chain_id: int) -> None:
        super().__init__()
        self._router = router
        self._chain_id = chain_id

    @property
    def endpoint_uri(self) -> str:
        return self._router._select_provider(self._chain_id).w3.provider.endpoint_uri

    def is_connected(self, show_traceback: bool = False) -> bool:
        try:
            return self._router._select_provider(
                self._chain_id
            ).w3.provider.is_connected(show_traceback)
        except Exception:
            return False

    def make_request(self, method: RPCEndpoint, params: Any) -> RPCResponse:
        attempts = len(self._router._providers.get(self._chain_id, [])) or 1
        last_exc: Exception | None = None
        for _ in range(attempts):
            state = self._router._select_provider(
                self._chain_id
            )  # raises ValueError if no providers
            try:
                return state.w3.provider.make_request(method, params)
            except _FAILOVER_EXC as exc:
                last_exc = exc
                self._router.report_failure(self._chain_id)
        assert last_exc is not None
        raise last_exc

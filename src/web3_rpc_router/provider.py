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

# HTTP statuses a request earns by its own content: a block the chain has not reached answers 400
# and an oversized batch 413. Every provider refuses these alike, so failing over would only
# demote each provider in turn; the error is raised as-is. Any other status is the provider
# refusing or failing (a bad, revoked or exhausted key, a rate limit, a server error).
_REQUEST_FAULT_STATUSES = (400, 413)


def _is_request_fault(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) in _REQUEST_FAULT_STATUSES


# web3's validation middleware reads eth_chainId before and after every eth_call, eth_estimateGas,
# eth_createAccessList and eth_sendTransaction, so without a cache one contract read costs three
# round trips. An endpoint's chain id never changes: each provider caches eth_chainId, and only
# that method. A threshold of None stores it without the extra eth_chainId request web3 makes to
# pick a block-validation threshold, which only methods other than eth_chainId use.
_CHAIN_ID_CACHE: dict[str, Any] = {
    "cache_allowed_requests": True,
    "cacheable_requests": {RPCEndpoint("eth_chainId")},
    "request_cache_validation_threshold": None,
}


@dataclass
class ChainConfig:
    """Health settings for one chain, overriding the router-wide values.

    A field left None takes the value given to ``RPCRouter``. Register with
    ``RPCRouter.configure_chain``; a chain without one uses the router-wide values throughout.
    """

    max_block_lag: int | None = None
    max_lag_seconds: float | None = None


@dataclass
class ProviderConfig:
    """Configuration for a single RPC provider."""

    name: str
    url: str
    priority: int  # Lower = higher priority (1 is best)
    request_timeout: int = 15


# Both providers are built with ``exception_retry_configuration=None``: one call is one HTTP
# request. web3's default resends a failed read to the same provider up to five times, with about
# 1.9s of backoff, on any HTTP or connection error. A rate limit, a rejected key or a server error
# would then reach RoutingProvider or a consumer's report_failure only after that delay, and a
# provider failing slowly enough would overrun the consumer's own timeout and never be reported.
# Where a failed request goes next is the router's decision: RoutingProvider sends it to the next
# provider, and get_web3 / get_async_web3 consumers retry after report_failure.


@dataclass
class ProviderState:
    """Runtime state for a provider (internal)."""

    config: ProviderConfig
    w3: Web3 = field(init=False, repr=False)
    async_w3: AsyncWeb3 = field(init=False, repr=False)
    # The pooled aiohttp session the health checker probes this provider over,
    # assigned by ``RPCRouter._init_keepalive_sessions``. It is the same session
    # ``async_w3`` sends real traffic over, so probes measure the path real
    # requests take and cost no extra connections. Probes speak JSON-RPC on it
    # directly rather than through ``async_w3``: web3 serialises every async
    # request through a process-wide ``threading.Lock`` acquired on an executor
    # thread, and a probe cancelled by its deadline while that acquire is in
    # flight would abandon the acquire and leave the lock held forever, wedging
    # all real traffic in the process. A plain aiohttp request has no such
    # cancellation hazard.
    probe_session: Any = field(default=None, repr=False)
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
    # Monotonic time of the last health-check answer (0 before the first), and the seconds
    # per block measured between two answers (0 until measured). The chain's block time is
    # the median of its providers' rates; see HealthChecker._allowed_lag.
    last_answer_at: float = 0.0
    seconds_per_block: float = 0.0

    def __post_init__(self) -> None:
        timeout = self.config.request_timeout
        self.w3 = Web3(
            Web3.HTTPProvider(
                self.config.url,
                request_kwargs={"timeout": timeout},
                exception_retry_configuration=None,
                **_CHAIN_ID_CACHE,
            )
        )
        self.async_w3 = AsyncWeb3(
            AsyncWeb3.AsyncHTTPProvider(
                self.config.url,
                request_kwargs={"timeout": timeout},
                exception_retry_configuration=None,
                **_CHAIN_ID_CACHE,
            )
        )


class RoutingProvider(JSONBaseProvider):
    """A sync web3 provider that sends each request to the RPCRouter's current-best provider for a
    chain, failing over to the next provider on transport errors.

    Lets a single `Web3` (built via `RPCRouter.web3(chain_id)`) transparently follow the router with
    no consumer call-site changes: every `w3.eth.*` request re-selects the healthy provider. An
    RPC-level ``{"error": ...}`` response is returned as-is (deterministic, not retried), and an
    HTTP 400 or 413 is raised as-is.
    """

    def __init__(self, router: RPCRouter, chain_id: int) -> None:
        super().__init__()
        self._router = router
        self._chain_id = chain_id

    @property
    def endpoint_uri(self) -> str:
        return self._router._select_provider(self._chain_id).w3.provider.endpoint_uri

    def is_connected(self, show_traceback: bool = False) -> bool:
        try:
            return self._router._select_provider(self._chain_id).w3.provider.is_connected(
                show_traceback
            )
        except ValueError:
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
                if _is_request_fault(exc):
                    raise
                last_exc = exc
                self._router.report_failure(self._chain_id, w3=state.w3)
        assert last_exc is not None
        raise last_exc

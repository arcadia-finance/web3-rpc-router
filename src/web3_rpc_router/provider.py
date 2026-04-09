from __future__ import annotations

from dataclasses import dataclass, field

from web3 import AsyncWeb3, Web3


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
    health_w3: AsyncWeb3 = field(init=False, repr=False)
    healthy: bool = True
    last_block: int = 0
    last_check: float = 0.0

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
        self.health_w3 = AsyncWeb3(
            AsyncWeb3.AsyncHTTPProvider(
                self.config.url,
                request_kwargs={"timeout": 5},
            )
        )

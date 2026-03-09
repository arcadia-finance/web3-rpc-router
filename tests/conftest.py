import pytest


@pytest.fixture
def make_router():
    """Factory to create a router with pre-configured providers."""
    from web3_rpc_router import RPCRouter

    def _make(providers_by_chain=None, **kwargs):
        router = RPCRouter(**kwargs)
        if providers_by_chain:
            for chain_id, configs in providers_by_chain.items():
                for config in configs:
                    router.add_provider(chain_id, config)
        return router

    return _make

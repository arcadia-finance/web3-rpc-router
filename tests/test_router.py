import pytest

from web3_rpc_router import RPCRouter, ProviderConfig
from web3_rpc_router.provider import ProviderState


class TestAddProvider:
    def test_adds_and_sorts_by_priority(self):
        router = RPCRouter()
        router.add_provider(1, ProviderConfig(name="low", url="http://a", priority=3))
        router.add_provider(1, ProviderConfig(name="high", url="http://b", priority=1))
        router.add_provider(1, ProviderConfig(name="mid", url="http://c", priority=2))

        names = [p.config.name for p in router._providers[1]]
        assert names == ["high", "mid", "low"]

    def test_separate_chains(self):
        router = RPCRouter()
        router.add_provider(1, ProviderConfig(name="a", url="http://a", priority=1))
        router.add_provider(2, ProviderConfig(name="b", url="http://b", priority=1))

        assert len(router._providers[1]) == 1
        assert len(router._providers[2]) == 1

    def test_chain_ids_property(self):
        router = RPCRouter()
        router.add_provider(8453, ProviderConfig(name="a", url="http://a", priority=1))
        router.add_provider(130, ProviderConfig(name="b", url="http://b", priority=1))

        assert set(router.chain_ids) == {8453, 130}


class TestGetWeb3:
    def _make_router_with_states(self, states):
        """Helper: create router and manually set provider states."""
        router = RPCRouter()
        router._providers[1] = states
        return router

    def test_returns_highest_priority_healthy(self):
        s1 = ProviderState(config=ProviderConfig(name="a", url="http://a", priority=1))
        s2 = ProviderState(config=ProviderConfig(name="b", url="http://b", priority=2))
        s1.healthy = True
        s2.healthy = True

        router = self._make_router_with_states([s1, s2])
        assert router.get_web3(1) is s1.w3

    def test_skips_unhealthy(self):
        s1 = ProviderState(config=ProviderConfig(name="a", url="http://a", priority=1))
        s2 = ProviderState(config=ProviderConfig(name="b", url="http://b", priority=2))
        s1.healthy = False
        s2.healthy = True

        router = self._make_router_with_states([s1, s2])
        assert router.get_web3(1) is s2.w3

    def test_degraded_mode_all_unhealthy(self):
        s1 = ProviderState(config=ProviderConfig(name="a", url="http://a", priority=1))
        s2 = ProviderState(config=ProviderConfig(name="b", url="http://b", priority=2))
        s1.healthy = False
        s2.healthy = False

        router = self._make_router_with_states([s1, s2])
        assert router.get_web3(1) is s1.w3

    def test_raises_for_unknown_chain(self):
        router = RPCRouter()
        with pytest.raises(ValueError, match="No providers configured"):
            router.get_web3(999)


class TestGetAsyncWeb3:
    def _make_router_with_states(self, states):
        router = RPCRouter()
        router._providers[1] = states
        return router

    def test_returns_async_web3_instance(self):
        s1 = ProviderState(config=ProviderConfig(name="a", url="http://a", priority=1))
        s1.healthy = True

        router = self._make_router_with_states([s1])
        assert router.get_async_web3(1) is s1.async_w3

    def test_skips_unhealthy(self):
        s1 = ProviderState(config=ProviderConfig(name="a", url="http://a", priority=1))
        s2 = ProviderState(config=ProviderConfig(name="b", url="http://b", priority=2))
        s1.healthy = False
        s2.healthy = True

        router = self._make_router_with_states([s1, s2])
        assert router.get_async_web3(1) is s2.async_w3

    def test_degraded_mode_all_unhealthy(self):
        s1 = ProviderState(config=ProviderConfig(name="a", url="http://a", priority=1))
        s2 = ProviderState(config=ProviderConfig(name="b", url="http://b", priority=2))
        s1.healthy = False
        s2.healthy = False

        router = self._make_router_with_states([s1, s2])
        assert router.get_async_web3(1) is s1.async_w3

    def test_raises_for_unknown_chain(self):
        router = RPCRouter()
        with pytest.raises(ValueError, match="No providers configured"):
            router.get_async_web3(999)


class TestGetProviderStatus:
    def test_returns_status_dicts(self):
        router = RPCRouter()
        router.add_provider(1, ProviderConfig(name="local", url="http://a", priority=1))
        router.add_provider(
            1, ProviderConfig(name="alchemy", url="http://b", priority=2)
        )

        status = router.get_provider_status(1)
        assert len(status) == 2
        assert status[0]["name"] == "local"
        assert status[1]["name"] == "alchemy"

    def test_empty_for_unknown_chain(self):
        router = RPCRouter()
        assert router.get_provider_status(999) == []

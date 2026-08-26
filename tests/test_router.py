import time

import aiohttp
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


class TestCooldownSelection:
    def _make_router(self, states):
        router = RPCRouter()
        router._providers[1] = states
        return router

    def test_skips_cooldown_provider(self):
        s1 = ProviderState(config=ProviderConfig(name="a", url="http://a", priority=1))
        s2 = ProviderState(config=ProviderConfig(name="b", url="http://b", priority=2))
        s1.healthy = True
        s2.healthy = True
        s1.cooldown_until = time.time() + 30

        router = self._make_router([s1, s2])
        assert router.get_async_web3(1) is s2.async_w3

    def test_prefers_fresh_over_cooldown(self):
        s1 = ProviderState(config=ProviderConfig(name="a", url="http://a", priority=1))
        s2 = ProviderState(config=ProviderConfig(name="b", url="http://b", priority=2))
        s3 = ProviderState(config=ProviderConfig(name="c", url="http://c", priority=3))
        s1.healthy = True
        s2.healthy = True
        s3.healthy = True
        s1.cooldown_until = time.time() + 30
        s2.cooldown_until = time.time() + 30

        router = self._make_router([s1, s2, s3])
        assert router.get_async_web3(1) is s3.async_w3

    def test_expired_cooldown_is_reused(self):
        s1 = ProviderState(config=ProviderConfig(name="a", url="http://a", priority=1))
        s2 = ProviderState(config=ProviderConfig(name="b", url="http://b", priority=2))
        s1.healthy = True
        s2.healthy = True
        s1.cooldown_until = time.time() - 1  # expired

        router = self._make_router([s1, s2])
        assert router.get_async_web3(1) is s1.async_w3

    def test_falls_back_to_cooldown_when_no_fresh_healthy(self):
        s1 = ProviderState(config=ProviderConfig(name="a", url="http://a", priority=1))
        s2 = ProviderState(config=ProviderConfig(name="b", url="http://b", priority=2))
        s1.healthy = True
        s2.healthy = True
        s1.cooldown_until = time.time() + 30
        s2.cooldown_until = time.time() + 30

        router = self._make_router([s1, s2])
        # Both are cooling down; highest-priority healthy still wins.
        assert router.get_async_web3(1) is s1.async_w3

    def test_unhealthy_ignored_even_if_not_cooling(self):
        s1 = ProviderState(config=ProviderConfig(name="a", url="http://a", priority=1))
        s2 = ProviderState(config=ProviderConfig(name="b", url="http://b", priority=2))
        s1.healthy = False  # dead
        s2.healthy = True

        router = self._make_router([s1, s2])
        assert router.get_async_web3(1) is s2.async_w3


class TestReportFailure:
    def _make_router(self, states):
        router = RPCRouter()
        router._providers[1] = states
        return router

    def test_demotes_currently_selected(self):
        s1 = ProviderState(config=ProviderConfig(name="a", url="http://a", priority=1))
        s2 = ProviderState(config=ProviderConfig(name="b", url="http://b", priority=2))
        s1.healthy = True
        s2.healthy = True

        router = self._make_router([s1, s2])
        router.report_failure(1, cooldown=30)

        assert s1.cooldown_until > time.time()
        assert s2.cooldown_until == 0.0
        # After the failure, next selection should rotate to s2.
        assert router.get_async_web3(1) is s2.async_w3

    def test_second_failure_demotes_next_provider(self):
        s1 = ProviderState(config=ProviderConfig(name="a", url="http://a", priority=1))
        s2 = ProviderState(config=ProviderConfig(name="b", url="http://b", priority=2))
        s3 = ProviderState(config=ProviderConfig(name="c", url="http://c", priority=3))
        s1.healthy = True
        s2.healthy = True
        s3.healthy = True

        router = self._make_router([s1, s2, s3])
        router.report_failure(1, cooldown=30)
        router.report_failure(1, cooldown=30)

        assert s1.cooldown_until > 0
        assert s2.cooldown_until > 0
        assert s3.cooldown_until == 0.0
        assert router.get_async_web3(1) is s3.async_w3

    def test_unknown_chain_is_noop(self):
        router = RPCRouter()
        # Should not raise when no providers are configured for the chain.
        router.report_failure(999)

    def test_custom_cooldown_duration(self):
        s1 = ProviderState(config=ProviderConfig(name="a", url="http://a", priority=1))
        s1.healthy = True

        router = self._make_router([s1])
        before = time.time()
        router.report_failure(1, cooldown=120)
        after = time.time()

        assert before + 120 <= s1.cooldown_until <= after + 120


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


class TestResolverChoice:
    """Sessions must get a resolver explicitly, and must still build without aiodns."""

    @pytest.mark.asyncio
    async def test_defaults_to_threaded_resolver(self):
        """The only resolver whose behaviour under concurrency is known here. A shared
        AsyncResolver funnels every lookup through one c-ares channel and degrades as
        concurrency rises, which lands on exactly the requests needing a new connection.
        """
        from web3_rpc_router.router import _build_resolver

        assert isinstance(_build_resolver(False), aiohttp.ThreadedResolver)

    @pytest.mark.asyncio
    async def test_async_dns_must_be_asked_for_explicitly(self):
        """Picking c-ares because aiodns became importable changes production DNS with no
        diff to review."""
        from web3_rpc_router.router import _build_resolver, aiodns_default

        resolver = _build_resolver(True)
        expected = aiohttp.AsyncResolver if aiodns_default else aiohttp.ThreadedResolver
        assert isinstance(resolver, expected)

    @pytest.mark.asyncio
    async def test_router_does_not_opt_into_async_dns_by_default(self):
        assert RPCRouter()._use_async_dns is False

    @pytest.mark.asyncio
    async def test_keepalive_sessions_get_a_resolver(self):
        router = RPCRouter()
        router.add_provider(1, ProviderConfig(name="a", url="http://a", priority=1))
        await router._init_keepalive_sessions()
        try:
            assert router._sessions
            for session in router._sessions:
                assert session.connector._resolver is not None
        finally:
            for session in router._sessions:
                await session.close()

    @pytest.mark.asyncio
    async def test_sessions_share_one_resolver_and_stop_closes_it(self):
        """Passing a resolver makes the connector a borrower, so the router must close
        it: aiohttp only closes a resolver it built itself.
        """
        router = RPCRouter()
        router.add_provider(1, ProviderConfig(name="a", url="http://a", priority=1))
        router.add_provider(1, ProviderConfig(name="b", url="http://b", priority=2))
        await router._init_keepalive_sessions()

        resolver = router._resolver
        assert resolver is not None
        assert len(router._sessions) == 2
        # One resolver, borrowed by every connector.
        for session in router._sessions:
            assert session.connector._resolver is resolver
            assert session.connector._resolver_owner is False

        sessions = list(router._sessions)  # stop() empties the list
        await router.stop()

        assert router._resolver is None
        assert sessions and all(s.closed for s in sessions)

    @pytest.mark.asyncio
    async def test_probes_run_on_the_shared_pooled_session(self):
        """The health checker probes over the same keep-alive session real traffic
        uses, so probes measure the real path and cost no extra connections. Real
        traffic keeps web3's retry policy; the probe is a single plain request.
        """
        router = RPCRouter()
        router.add_provider(1, ProviderConfig(name="a", url="http://a", priority=1))

        await router._init_keepalive_sessions()
        try:
            state = router._providers[1][0]
            assert state.probe_session is router._sessions[0]
            assert state.async_w3.provider.exception_retry_configuration is not None
        finally:
            await router.stop()

    @pytest.mark.asyncio
    async def test_restart_keeps_the_routers_own_pooled_session(self):
        """cache_async_session honours the session it is given only on a cache miss, so
        a second start() would otherwise get a web3-built force_close session instead.
        """
        router = RPCRouter()
        router.add_provider(1, ProviderConfig(name="a", url="http://a", priority=1))

        await router._init_keepalive_sessions()
        first = router._sessions[0]
        provider = router._providers[1][0]
        assert provider.async_w3.provider._request_session_manager.session_cache._data
        await router.stop()

        await router._init_keepalive_sessions()
        try:
            second = router._sessions[0]
            assert second is not first
            cached = list(
                provider.async_w3.provider._request_session_manager.session_cache._data.values()
            )
            assert cached == [second]
            assert second.connector._resolver is router._resolver
            assert second.connector._force_close is False
        finally:
            await router.stop()

    @pytest.mark.asyncio
    async def test_a_session_is_tracked_before_it_is_handed_to_web3(self):
        """A failure while handing the session over must not leak it."""
        router = RPCRouter()
        router.add_provider(1, ProviderConfig(name="a", url="http://a", priority=1))
        provider = router._providers[1][0]

        async def boom(_session):
            raise RuntimeError("handover failed")

        provider.async_w3.provider.cache_async_session = boom
        with pytest.raises(RuntimeError):
            await router._init_keepalive_sessions()

        assert len(router._sessions) == 1  # tracked, so stop() can close it
        await router.stop()
        assert router._resolver is None


class TestDegradedModeUsesLastKnownGood:
    def test_falls_back_to_the_provider_that_last_proved_it_works(self):
        """The highest-priority provider is often the one whose failure got us here, so
        priority order is the wrong fallback when nothing is healthy."""
        router = RPCRouter()
        router.add_provider(
            1, ProviderConfig(name="primary", url="http://a", priority=1)
        )
        router.add_provider(
            1, ProviderConfig(name="backup", url="http://b", priority=2)
        )
        primary, backup = router._providers[1]
        primary.healthy = backup.healthy = False
        primary.last_block = 100  # went dark a while ago
        backup.last_block = 5_000  # answered much more recently

        assert router._select_provider(1) is backup

    def test_ties_and_cold_start_fall_back_to_priority(self):
        router = RPCRouter()
        router.add_provider(
            1, ProviderConfig(name="primary", url="http://a", priority=1)
        )
        router.add_provider(
            1, ProviderConfig(name="backup", url="http://b", priority=2)
        )
        for p in router._providers[1]:
            p.healthy = False  # nothing has ever answered, last_block == 0

        assert router._select_provider(1).config.name == "primary"

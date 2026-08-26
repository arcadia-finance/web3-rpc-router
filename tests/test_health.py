import asyncio
import time

import pytest

from web3_rpc_router.provider import ProviderConfig, ProviderState
from web3_rpc_router.health import HealthChecker


class _FakeAsyncEth:
    """Stand-in for `probe_w3.eth` whose awaitable `block_number` resolves or raises.

    `HealthChecker._check_one` awaits `p.probe_w3.eth.block_number`, so the probe is
    stubbed on the probe client. `delay` simulates a provider that answers too slowly.
    """

    def __init__(self, block_number=None, error=None, delay=0.0):
        self._block_number = block_number
        self._error = error
        self._delay = delay
        self.calls = 0

    @property
    def block_number(self):
        self.calls += 1

        async def _probe():
            if self._delay:
                await asyncio.sleep(self._delay)
            if self._error is not None:
                raise self._error
            return self._block_number

        return _probe()


class _FakeAsyncW3:
    def __init__(self, eth):
        self.eth = eth


def _make_provider(name, priority=1, block_number=100, delay=0.0):
    """Create a ProviderState whose probe returns `block_number`."""
    state = ProviderState(
        config=ProviderConfig(name=name, url="http://fake", priority=priority)
    )
    state.probe_w3 = _FakeAsyncW3(_FakeAsyncEth(block_number=block_number, delay=delay))
    return state


def _fail_provider(state, error=None):
    """Make a provider's health check fail."""
    state.probe_w3 = _FakeAsyncW3(_FakeAsyncEth(error=error or ConnectionError("down")))


def _set_block(state, block_number):
    """Update a provider's mock block number."""
    state.probe_w3 = _FakeAsyncW3(_FakeAsyncEth(block_number=block_number))


class TestCheckAll:
    @pytest.mark.asyncio
    async def test_all_in_sync(self):
        p1 = _make_provider("a", block_number=100)
        p2 = _make_provider("b", block_number=100)
        providers = {1: [p1, p2]}

        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=5)
        await checker.check_all()

        assert p1.healthy is True
        assert p2.healthy is True
        assert p1.last_block == 100
        assert p2.last_block == 100

    @pytest.mark.asyncio
    async def test_one_lagging(self):
        p1 = _make_provider("a", block_number=100)
        p2 = _make_provider("b", block_number=97)
        providers = {1: [p1, p2]}

        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=5)
        await checker.check_all()

        assert p1.healthy is True
        assert p2.healthy is False

    @pytest.mark.asyncio
    async def test_lag_within_tolerance(self):
        p1 = _make_provider("a", block_number=100)
        p2 = _make_provider("b", block_number=99)
        providers = {1: [p1, p2]}

        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=5)
        await checker.check_all()

        assert p1.healthy is True
        assert p2.healthy is True  # 1 block behind, within tolerance

    @pytest.mark.asyncio
    async def test_provider_failure_marks_unhealthy(self):
        p1 = _make_provider("a", block_number=100)
        p2 = _make_provider("b", block_number=100)
        _fail_provider(p2)
        providers = {1: [p1, p2]}

        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=5)
        await checker.check_all()

        assert p1.healthy is True
        assert p2.healthy is False
        assert p2.last_block == 0

    @pytest.mark.asyncio
    async def test_provider_recovers(self):
        p1 = _make_provider("a", block_number=100)
        p2 = _make_provider("b", block_number=100)
        providers = {1: [p1, p2]}

        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=5)

        # First: p2 fails
        _fail_provider(p2)
        await checker.check_all()
        assert p2.healthy is False

        # Second: p2 recovers
        _set_block(p1, 101)
        _set_block(p2, 101)
        await checker.check_all()
        assert p2.healthy is True

    @pytest.mark.asyncio
    async def test_multiple_chains(self):
        p1 = _make_provider("a", block_number=100)
        p2 = _make_provider("b", block_number=200)
        providers = {1: [p1], 2: [p2]}

        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=5)
        await checker.check_all()

        # Each chain evaluated independently
        assert p1.healthy is True
        assert p1.last_block == 100
        assert p2.healthy is True
        assert p2.last_block == 200

    @pytest.mark.asyncio
    async def test_all_providers_fail(self):
        p1 = _make_provider("a")
        p2 = _make_provider("b")
        _fail_provider(p1)
        _fail_provider(p2)
        providers = {1: [p1, p2]}

        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=5)
        await checker.check_all()

        assert p1.healthy is False
        assert p2.healthy is False

    @pytest.mark.asyncio
    async def test_last_check_updated(self):
        p1 = _make_provider("a", block_number=100)
        providers = {1: [p1]}

        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=5)
        assert p1.last_check == 0.0

        await checker.check_all()
        assert p1.last_check > 0


class TestRetryInterval:
    @pytest.mark.asyncio
    async def test_has_unhealthy_when_provider_down(self):
        p1 = _make_provider("a", block_number=100)
        p2 = _make_provider("b", block_number=100)
        _fail_provider(p2)
        providers = {1: [p1, p2]}

        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=5)
        await checker.check_all()

        assert checker._has_unhealthy() is True

    @pytest.mark.asyncio
    async def test_has_unhealthy_false_when_all_healthy(self):
        p1 = _make_provider("a", block_number=100)
        p2 = _make_provider("b", block_number=100)
        providers = {1: [p1, p2]}

        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=5)
        await checker.check_all()

        assert checker._has_unhealthy() is False

    @pytest.mark.asyncio
    async def test_loop_uses_retry_interval_when_unhealthy(self):
        """When a provider is unhealthy, the loop should sleep retry_interval, not interval."""
        p1 = _make_provider("a", block_number=100)
        p2 = _make_provider("b", block_number=100)
        _fail_provider(p2)
        providers = {1: [p1, p2]}

        checker = HealthChecker(
            providers,
            interval=600,
            max_block_lag=1,
            timeout=5,
            retry_interval=0.05,
        )
        await checker.check_all()
        assert p2.healthy is False

        # Fix p2 and start the loop — it should re-check within retry_interval
        _set_block(p2, 100)
        checker.start()
        await asyncio.sleep(0.15)  # 3x retry_interval to be safe
        checker.stop()

        assert p2.healthy is True

    @pytest.mark.asyncio
    async def test_loop_uses_full_interval_when_all_healthy(self):
        """When all providers are healthy, the loop should sleep the full interval."""
        p1 = _make_provider("a", block_number=100)
        providers = {1: [p1]}

        checker = HealthChecker(
            providers,
            interval=600,
            max_block_lag=1,
            timeout=5,
            retry_interval=0.05,
        )
        await checker.check_all()
        first_check = p1.last_check

        # Start the loop — it should NOT re-check within retry_interval
        checker.start()
        await asyncio.sleep(0.15)
        checker.stop()

        # last_check should be unchanged since interval=600 hasn't elapsed
        assert p1.last_check == first_check


class TestCooldownReset:
    """A successful health check should clear the request-level cooldown so the
    provider becomes eligible for selection again without waiting for the
    cooldown timer to expire naturally.
    """

    @pytest.mark.asyncio
    async def test_successful_check_clears_cooldown(self, monkeypatch):
        p1 = ProviderState(
            config=ProviderConfig(name="a", url="http://fake", priority=1)
        )
        p1.cooldown_until = time.time() + 120  # demoted
        providers = {1: [p1]}

        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=5)

        # Stub the probe itself; this test only covers the post-success bookkeeping.
        async def ok(_p):
            return 100

        monkeypatch.setattr(checker, "_check_one", ok)

        await checker.check_all()

        assert p1.cooldown_until == 0.0
        assert p1.healthy is True

    @pytest.mark.asyncio
    async def test_failed_check_preserves_cooldown(self, monkeypatch):
        p1 = ProviderState(
            config=ProviderConfig(name="a", url="http://fake", priority=1)
        )
        target = time.time() + 120
        p1.cooldown_until = target
        providers = {1: [p1]}

        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=5)

        async def boom(_p):
            raise ConnectionError("down")

        monkeypatch.setattr(checker, "_check_one", boom)

        await checker.check_all()

        # Health check failed, so cooldown must NOT be reset — the provider
        # should still be treated as demoted by the router.
        assert p1.cooldown_until == target


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_creates_task(self):
        providers = {1: [_make_provider("a")]}
        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=5)

        checker.start()
        assert checker._task is not None
        assert not checker._task.done()

        checker.stop()
        # Give the event loop a tick to process cancellation
        await asyncio.sleep(0)
        assert checker._task is None


class TestProbeIsolation:
    """One hanging provider must not affect any other provider's verdict.

    The probe is awaited on the event loop, so exceeding the timeout costs
    nothing beyond that provider's own result.
    """

    @pytest.mark.asyncio
    async def test_hanging_provider_does_not_starve_the_others(self):
        slow = _make_provider("slow", block_number=100, delay=30)
        fast = [_make_provider(f"fast{i}", block_number=100) for i in range(24)]
        providers = {1: [slow, *fast]}

        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=0.2)
        await asyncio.wait_for(checker.check_all(), timeout=5)

        assert slow.healthy is False
        assert all(p.healthy for p in fast)
        assert all(p.last_block == 100 for p in fast)

    @pytest.mark.asyncio
    async def test_probe_does_not_use_the_default_executor(self):
        """asyncio's default executor is shared with aiohttp's DNS resolution and
        response decompression. A probe that occupied a worker there could not be
        released by the timeout, so the probe must never touch it.
        """
        providers = {1: [_make_provider("a", block_number=100, delay=30)]}
        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=0.05)

        loop = asyncio.get_running_loop()
        real_run_in_executor = loop.run_in_executor
        offloaded = []

        def spy(executor, func, *args):
            if executor is None:
                offloaded.append(func)
            return real_run_in_executor(executor, func, *args)

        loop.run_in_executor = spy
        try:
            await checker.check_all()
        finally:
            loop.run_in_executor = real_run_in_executor

        assert offloaded == []


class TestProbeBackoff:
    """A provider that stays down must be probed progressively less often.

    Every provider is otherwise probed on every cycle, so a permanently dead
    endpoint would be polled for as long as it stays configured.
    """

    def test_backoff_delay_progression(self):
        checker = HealthChecker(
            {}, interval=60, max_block_lag=1, timeout=5, retry_interval=30
        )
        assert checker._backoff_delay(0) == 0.0
        assert checker._backoff_delay(2) == 0.0  # still within its allowance
        assert checker._backoff_delay(3) == 30.0
        assert checker._backoff_delay(4) == 60.0
        assert checker._backoff_delay(5) == 120.0
        assert checker._backoff_delay(99) == 300.0  # capped

    @pytest.mark.asyncio
    async def test_dead_provider_is_skipped_once_backing_off(self):
        dead = _make_provider("dead")
        _fail_provider(dead)
        live = _make_provider("live", block_number=100)
        providers = {1: [dead, live]}
        checker = HealthChecker(
            providers, interval=60, max_block_lag=1, timeout=5, retry_interval=30
        )

        for _ in range(3):
            await checker.check_all()

        assert dead.consecutive_failures == 3
        assert dead.healthy is False
        assert dead.next_check > time.time()
        probes_when_backoff_began = dead.probe_w3.eth.calls

        # Further cycles must leave it alone while its window is open...
        await checker.check_all()
        await checker.check_all()
        assert dead.probe_w3.eth.calls == probes_when_backoff_began
        # ...while the healthy provider is still checked every cycle.
        assert live.probe_w3.eth.calls == 5
        assert live.healthy is True

    @pytest.mark.asyncio
    async def test_recovery_clears_the_backoff(self):
        p1 = _make_provider("a")
        _fail_provider(p1)
        providers = {1: [p1]}
        checker = HealthChecker(
            providers, interval=60, max_block_lag=1, timeout=5, retry_interval=30
        )

        for _ in range(3):
            await checker.check_all()
        assert p1.next_check > 0

        _set_block(p1, 100)
        # No manual reset: p1 is the only provider and it is unhealthy, so the chain
        # counts as dark and is probed regardless of its window.
        await checker.check_all()

        assert p1.healthy is True
        assert p1.consecutive_failures == 0
        assert p1.next_check == 0.0

    @pytest.mark.asyncio
    async def test_cycle_is_a_noop_when_every_provider_is_backing_off(self):
        p1 = _make_provider("a", block_number=100)
        p1.next_check = time.time() + 300
        providers = {1: [p1]}
        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=5)

        await checker.check_all()

        assert p1.probe_w3.eth.calls == 0
        assert p1.last_check == 0.0

    @pytest.mark.asyncio
    async def test_backoff_is_ignored_while_a_chain_has_no_healthy_provider(self):
        """A dark chain is served in degraded mode, so noticing a recovery beats
        sparing a dead endpoint: every provider is probed regardless of its window.
        """
        a = _make_provider("a")
        b = _make_provider("b")
        _fail_provider(a)
        _fail_provider(b)
        providers = {1: [a, b]}
        checker = HealthChecker(
            providers, interval=60, max_block_lag=1, timeout=5, retry_interval=30
        )

        for _ in range(3):
            await checker.check_all()

        assert a.healthy is False and b.healthy is False
        assert a.next_check > time.time()  # a backoff window is open...
        probes = a.probe_w3.eth.calls

        await checker.check_all()  # ...but the chain has nothing healthy left

        assert a.probe_w3.eth.calls == probes + 1

    @pytest.mark.asyncio
    async def test_backoff_still_applies_while_the_chain_has_a_healthy_provider(self):
        """The dark-chain exception must not defeat the backoff in the normal case."""
        dead = _make_provider("dead")
        _fail_provider(dead)
        live = _make_provider("live", block_number=100)
        providers = {1: [dead, live]}
        checker = HealthChecker(
            providers, interval=60, max_block_lag=1, timeout=5, retry_interval=30
        )

        for _ in range(3):
            await checker.check_all()
        probes = dead.probe_w3.eth.calls

        await checker.check_all()

        assert live.healthy is True
        assert dead.probe_w3.eth.calls == probes


class TestNonBlockResults:
    """Anything a probe returns that is not a block number must count as a failure."""

    @pytest.mark.asyncio
    async def test_cancelled_probe_is_a_failure_not_a_block_number(self, monkeypatch):
        """asyncio.gather hands back a CancelledError as a *result*, and it derives from
        BaseException, so an ``isinstance(result, Exception)`` test would store it as
        last_block and make every later max() over the chain's blocks raise.
        """
        p1 = _make_provider("a", block_number=100)
        providers = {1: [p1]}
        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=5)

        async def cancelled(_p):
            raise asyncio.CancelledError()

        monkeypatch.setattr(checker, "_check_one", cancelled)
        await checker.check_all()

        assert p1.last_block == 0
        assert p1.healthy is False
        assert p1.consecutive_failures == 1
        # The chain stays arithmetically usable.
        assert max(q.last_block for q in providers[1]) == 0


class TestCooldownRace:
    @pytest.mark.asyncio
    async def test_a_demotion_during_an_in_flight_probe_survives_it(self):
        """report_failure() can demote a provider while its probe is in flight. A
        success that started before the demotion must not clear it, or the router
        re-selects the endpoint a real request just failed on.
        """
        p1 = _make_provider("a", block_number=100, delay=0.05)
        providers = {1: [p1]}
        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=5)

        cycle = asyncio.ensure_future(checker.check_all())
        await asyncio.sleep(0.01)  # probe is in flight
        p1.cooldown_until = time.time() + 60  # as report_failure() would
        await cycle

        assert p1.cooldown_until > time.time()
        assert p1.healthy is True  # the probe still counted as a success

    @pytest.mark.asyncio
    async def test_a_stale_cooldown_is_still_cleared_by_a_success(self):
        """The guard must not stop a success from clearing a cooldown it did race."""
        p1 = _make_provider("a", block_number=100)
        p1.cooldown_until = time.time() + 60
        providers = {1: [p1]}
        checker = HealthChecker(providers, interval=60, max_block_lag=1, timeout=5)

        await checker.check_all()

        assert p1.cooldown_until == 0.0


class TestCadenceVsBackoff:
    """A dead provider must not hold every chain at the fast retry cadence."""

    @pytest.mark.asyncio
    async def test_a_backed_off_provider_releases_the_fast_cadence(self):
        dead = _make_provider("dead")
        _fail_provider(dead)
        live = _make_provider("live", block_number=100)
        providers = {1: [dead, live]}
        checker = HealthChecker(
            providers, interval=60, max_block_lag=1, timeout=5, retry_interval=30
        )

        for _ in range(3):
            await checker.check_all()

        assert dead.healthy is False
        assert dead.next_check > time.time()
        # Its window is open, so nothing is waiting on it: use the full interval.
        assert checker._has_unhealthy() is False

    @pytest.mark.asyncio
    async def test_the_fast_cadence_returns_once_the_window_expires(self):
        dead = _make_provider("dead")
        _fail_provider(dead)
        live = _make_provider("live", block_number=100)
        providers = {1: [dead, live]}
        checker = HealthChecker(
            providers, interval=60, max_block_lag=1, timeout=5, retry_interval=30
        )

        for _ in range(3):
            await checker.check_all()
        assert checker._has_unhealthy() is False

        dead.next_check = 0.0  # window elapsed, it is due again
        assert checker._has_unhealthy() is True

    @pytest.mark.asyncio
    async def test_a_dark_chain_keeps_the_fast_cadence(self):
        """Its providers are probed regardless of their windows, so the cadence must
        match: this is the case where recovery matters most.
        """
        a = _make_provider("a")
        b = _make_provider("b")
        _fail_provider(a)
        _fail_provider(b)
        providers = {1: [a, b]}
        checker = HealthChecker(
            providers, interval=60, max_block_lag=1, timeout=5, retry_interval=30
        )

        for _ in range(3):
            await checker.check_all()

        assert a.next_check > time.time() and b.next_check > time.time()
        assert checker._has_unhealthy() is True

    @pytest.mark.asyncio
    async def test_a_lagging_but_responsive_provider_keeps_the_fast_cadence(self):
        """It answers, so it never backs off, and it may yet catch up."""
        ahead = _make_provider("ahead", block_number=100)
        behind = _make_provider("behind", block_number=50)
        providers = {1: [ahead, behind]}
        checker = HealthChecker(
            providers, interval=60, max_block_lag=1, timeout=5, retry_interval=30
        )

        await checker.check_all()

        assert behind.healthy is False
        assert behind.consecutive_failures == 0 and behind.next_check == 0.0
        assert checker._has_unhealthy() is True

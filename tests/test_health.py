import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from web3_rpc_router.provider import ProviderConfig, ProviderState
from web3_rpc_router.health import HealthChecker


def _make_provider(name, priority=1, block_number=100):
    """Create a ProviderState with a mocked async_w3.eth.get_block_number."""
    state = ProviderState(
        config=ProviderConfig(name=name, url="http://fake", priority=priority)
    )
    # Mock the async web3 used by the health checker
    state.async_w3.eth.get_block_number = AsyncMock(return_value=block_number)
    return state


def _fail_provider(state, error=None):
    """Make a provider's health check fail."""
    state.async_w3.eth.get_block_number = AsyncMock(
        side_effect=error or ConnectionError("down")
    )


def _set_block(state, block_number):
    """Update a provider's mock block number."""
    state.async_w3.eth.get_block_number = AsyncMock(return_value=block_number)


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
            providers, interval=600, max_block_lag=1, timeout=5, retry_interval=0.05,
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
            providers, interval=600, max_block_lag=1, timeout=5, retry_interval=0.05,
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
        # Bypass the broken thread-based block_number probe by stubbing
        # _check_one directly — we only care about the post-success bookkeeping.
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

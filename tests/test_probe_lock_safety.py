"""Probes must be safe to cancel while web3's session-manager lock is contended.

web3 serialises every async request through ``HTTPSessionManager._lock``, one
``threading.Lock`` shared by every provider in the process, acquired on an
executor thread. A task cancelled while that acquire is in flight abandons the
acquire: it completes in the executor with no owner left to release it, and the
lock stays held forever — every later async web3 request in the process then
blocks behind it until its caller's deadline. Health probes are cancelled by
design (``asyncio.wait_for``) whenever a provider or a CPU-throttled instance is
slow, so a probe that took that lock would sooner or later wedge all real
traffic. These tests pin the property that probes never touch it.
"""

import asyncio
import threading

import pytest
from aiohttp import web
from web3._utils.http_session_manager import HTTPSessionManager

from web3_rpc_router import ProviderConfig, RPCRouter


@pytest.fixture(autouse=True)
def fresh_web3_lock():
    """Isolate the process-wide lock so a failing run cannot wedge the suite."""
    original = HTTPSessionManager._lock
    HTTPSessionManager._lock = threading.Lock()
    yield
    HTTPSessionManager._lock = original


async def _start_rpc_server():
    async def handler(request):
        body = await request.json()
        return web.json_response(
            {"jsonrpc": "2.0", "id": body.get("id", 1), "result": hex(1_000_000)}
        )

    app = web.Application()
    app.router.add_post("/{tail:.*}", handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, port


async def _stop_router(router):
    try:
        await asyncio.wait_for(router.stop(), timeout=3)
    except asyncio.TimeoutError:
        pass  # a wedged run must still let the test report its failure


@pytest.mark.asyncio
async def test_probe_cycle_survives_a_contended_web3_lock():
    """The production collapse, distilled: the session-manager lock is contended
    (a request holds it across a loop stall) exactly when a probe cycle's
    deadlines expire. Probes must still succeed — they do not need that lock —
    and afterwards the lock must be exactly as its holder left it, with real
    web3 traffic flowing once released.
    """
    runner, port = await _start_rpc_server()
    router = RPCRouter(check_interval=3600, max_block_lag=5, health_check_timeout=0.8)
    for i in range(8):
        router.add_provider(
            1,
            ProviderConfig(
                name=f"p{i}", url=f"http://127.0.0.1:{port}/{i}", priority=i + 1
            ),
        )
    try:
        await router.start()

        HTTPSessionManager._lock.acquire()  # a real request mid-critical-section
        try:
            await router._health_checker.check_all()
        finally:
            still_ours = HTTPSessionManager._lock.locked()
            if still_ours:
                HTTPSessionManager._lock.release()

        providers = router._providers[1]
        assert all(
            p.consecutive_failures == 0 for p in providers
        ), "probes must not block on web3's session-manager lock"
        assert all(p.healthy for p in providers)
        assert all(p.last_block == 1_000_000 for p in providers)
        assert still_ours, "a probe released a lock it did not own"

        await asyncio.sleep(0.3)  # room for any abandoned acquire to land
        assert (
            not HTTPSessionManager._lock.locked()
        ), "a cancelled probe left web3's process-wide lock held"

        block = await asyncio.wait_for(
            router.get_async_web3(1).eth.block_number, timeout=5
        )
        assert block == 1_000_000
    finally:
        await _stop_router(router)
        await runner.cleanup()


@pytest.mark.asyncio
async def test_probe_cycle_never_acquires_the_web3_lock():
    """Pin the mechanism itself: a full probe cycle performs zero acquisitions of
    web3's session-manager lock."""

    class _CountingLock:
        def __init__(self):
            self._lock = threading.Lock()
            self.acquires = 0

        def acquire(self, *a, **kw):
            self.acquires += 1
            return self._lock.acquire(*a, **kw)

        def release(self):
            self._lock.release()

        def locked(self):
            return self._lock.locked()

    counting = _CountingLock()
    HTTPSessionManager._lock = counting

    runner, port = await _start_rpc_server()
    router = RPCRouter(check_interval=3600, max_block_lag=5, health_check_timeout=2)
    for i in range(4):
        router.add_provider(
            1,
            ProviderConfig(
                name=f"p{i}", url=f"http://127.0.0.1:{port}/{i}", priority=i + 1
            ),
        )
    try:
        await router.start()  # seeding the sessions may legitimately take the lock
        baseline = counting.acquires
        await router._health_checker.check_all()
        await router._health_checker.check_all()
        assert (
            counting.acquires == baseline
        ), "the probe path must stay off web3's session-manager lock"

        # The lock still works for real traffic, which is allowed to take it.
        block = await asyncio.wait_for(
            router.get_async_web3(1).eth.block_number, timeout=5
        )
        assert block == 1_000_000
        assert counting.acquires > 0
    finally:
        await _stop_router(router)
        await runner.cleanup()

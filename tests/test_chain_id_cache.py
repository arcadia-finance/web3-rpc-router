"""eth_chainId is read once per provider: web3's validation middleware asks for it around every
eth_call, which would otherwise triple the round trips of each contract read."""

import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import pytest
from web3 import AsyncWeb3

from web3_rpc_router import ProviderConfig, RPCRouter

CALL = {"to": "0x000000000000000000000000000000000000dEaD", "data": "0x"}
RESULTS = {"eth_chainId": "0x1", "eth_call": "0x" + "00" * 32, "eth_blockNumber": "0x10"}


@pytest.fixture
def node():
    """A local JSON-RPC node that counts the methods it receives."""
    seen = Counter()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen[body["method"]] += 1
            reply = json.dumps(
                {"jsonrpc": "2.0", "id": body["id"], "result": RESULTS[body["method"]]}
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply.encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", seen
    server.shutdown()
    server.server_close()
    thread.join()


def router_for(url):
    router = RPCRouter()
    router.add_provider(1, ProviderConfig(name="node", url=url, priority=1))
    return router


def test_sync_reads_ask_the_chain_id_once(node):
    url, seen = node
    router = router_for(url)

    for _ in range(5):
        router.web3(1).eth.call(CALL)
        router.get_web3(1).eth.call(CALL)

    assert seen["eth_call"] == 10
    assert seen["eth_chainId"] == 1


@pytest.mark.asyncio
async def test_async_reads_ask_the_chain_id_once(node):
    url, seen = node
    router = router_for(url)
    await router._init_keepalive_sessions()
    try:
        w3 = router.get_async_web3(1)
        for _ in range(5):
            await w3.eth.call(CALL)
        await w3.eth.block_number
    finally:
        await router.stop()

    assert seen["eth_call"] == 5
    assert seen["eth_blockNumber"] == 1
    assert seen["eth_chainId"] == 1


@pytest.mark.asyncio
async def test_only_the_chain_id_is_cached(node):
    url, seen = node
    router = router_for(url)
    await router._init_keepalive_sessions()
    try:
        w3 = router.get_async_web3(1)
        for _ in range(3):
            await w3.eth.call(CALL)
            await w3.eth.block_number
    finally:
        await router.stop()

    assert seen["eth_call"] == 3
    assert seen["eth_blockNumber"] == 3


@pytest.mark.asyncio
async def test_an_uncached_provider_asks_twice_per_read(node):
    """The behaviour the cache removes: web3's defaults request eth_chainId before and after each call."""
    url, seen = node
    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(url))
    session = aiohttp.ClientSession()
    await w3.provider.cache_async_session(session)
    try:
        for _ in range(5):
            await w3.eth.call(CALL)
    finally:
        await session.close()

    assert seen["eth_call"] == 5
    assert seen["eth_chainId"] == 10

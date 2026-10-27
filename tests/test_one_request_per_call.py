"""A failed request reaches the router after one HTTP request: web3 does not resend it to the same
provider first, so a refusing provider is rotated away from immediately."""

import json
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import pytest
import requests

from web3_rpc_router import ProviderConfig, RPCRouter

CALL = {"to": "0x000000000000000000000000000000000000dEaD", "data": "0x"}
RESULTS = {"eth_chainId": "0x1", "eth_call": "0x" + "00" * 32, "eth_blockNumber": "0x10"}


def _serve(call_status):
    """A local JSON-RPC node that answers ``eth_call`` with ``call_status`` and counts methods."""
    seen = Counter()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            method = body["method"]
            seen[method] += 1
            status = call_status if method == "eth_call" else 200
            if status == 200:
                payload = {"jsonrpc": "2.0", "id": body["id"], "result": RESULTS[method]}
            else:
                payload = {"jsonrpc": "2.0", "id": body["id"], "error": {"code": -32005}}
            reply = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def close():
        server.shutdown()
        server.server_close()
        thread.join()

    return f"http://127.0.0.1:{server.server_port}", seen, close


@pytest.fixture
def nodes():
    closers = []

    def _make(call_status):
        url, seen, close = _serve(call_status)
        closers.append(close)
        return url, seen

    yield _make
    for close in closers:
        close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429, 503])
async def test_async_failure_is_one_request(nodes, status):
    url, seen = nodes(status)
    router = RPCRouter()
    router.add_provider(1, ProviderConfig(name="node", url=url, priority=1))
    await router._init_keepalive_sessions()
    try:
        started = time.monotonic()
        with pytest.raises(aiohttp.ClientResponseError) as excinfo:
            await router.get_async_web3(1).eth.call(CALL)
        elapsed = time.monotonic() - started
    finally:
        await router.stop()

    assert excinfo.value.status == status
    assert seen["eth_call"] == 1
    assert elapsed < 0.5  # no backoff before the caller sees it


@pytest.mark.parametrize("status", [401, 429, 503])
def test_sync_failure_is_one_request(nodes, status):
    url, seen = nodes(status)
    router = RPCRouter()
    router.add_provider(1, ProviderConfig(name="node", url=url, priority=1))

    with pytest.raises(requests.HTTPError):
        router.get_web3(1).eth.call(CALL)

    assert seen["eth_call"] == 1


def test_routing_provider_moves_on_after_one_request(nodes):
    """The first provider rate limits reads: one request there, then the next provider answers."""
    limited_url, limited = nodes(429)
    healthy_url, healthy = nodes(200)
    router = RPCRouter()
    router.add_provider(1, ProviderConfig(name="limited", url=limited_url, priority=1))
    router.add_provider(1, ProviderConfig(name="healthy", url=healthy_url, priority=2))

    started = time.monotonic()
    result = router.web3(1).eth.call(CALL)
    elapsed = time.monotonic() - started

    assert result == bytes(32)
    assert limited["eth_call"] == 1
    assert healthy["eth_call"] == 1
    assert router._providers[1][0].cooldown_until > time.time()
    assert elapsed < 0.5

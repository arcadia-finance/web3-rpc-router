from unittest.mock import MagicMock

import pytest
import requests
from web3.exceptions import ProviderConnectionError

from web3_rpc_router import ProviderConfig, RoutingProvider, RPCRouter


def _router_two_providers():
    router = RPCRouter()
    router.add_provider(1, ProviderConfig(name="a", url="http://a", priority=1))
    router.add_provider(1, ProviderConfig(name="b", url="http://b", priority=2))
    return router


def test_failover_on_transport_error():
    """Provider A raises a transport error -> report_failure demotes it -> B serves the retry."""
    router = _router_two_providers()
    a, b = router._providers[1]
    a.w3.provider.make_request = MagicMock(
        side_effect=requests.exceptions.ConnectionError("A down")
    )
    b.w3.provider.make_request = MagicMock(
        return_value={"jsonrpc": "2.0", "id": 1, "result": "0x1"}
    )

    resp = RoutingProvider(router, 1).make_request("eth_blockNumber", [])

    assert resp["result"] == "0x1"
    a.w3.provider.make_request.assert_called_once()
    b.w3.provider.make_request.assert_called_once()
    assert a.cooldown_until > 0  # A was demoted by report_failure


def test_rpc_error_returned_without_retry():
    """An RPC-level {'error': ...} response is deterministic: return it, don't fail over."""
    router = _router_two_providers()
    a, b = router._providers[1]
    a.w3.provider.make_request = MagicMock(
        return_value={
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32000, "message": "x"},
        }
    )
    b.w3.provider.make_request = MagicMock(
        return_value={"jsonrpc": "2.0", "id": 1, "result": "0x1"}
    )

    resp = RoutingProvider(router, 1).make_request("eth_call", [])

    assert "error" in resp
    a.w3.provider.make_request.assert_called_once()
    b.w3.provider.make_request.assert_not_called()


def test_all_providers_exhausted_reraises():
    router = _router_two_providers()
    a, b = router._providers[1]
    a.w3.provider.make_request = MagicMock(side_effect=requests.exceptions.ConnectionError("A"))
    b.w3.provider.make_request = MagicMock(side_effect=requests.exceptions.ConnectionError("B"))

    with pytest.raises(requests.exceptions.ConnectionError):
        RoutingProvider(router, 1).make_request("eth_blockNumber", [])


def test_no_providers_raises_value_error():
    with pytest.raises(ValueError):
        RoutingProvider(RPCRouter(), 1).make_request("eth_blockNumber", [])


def test_router_web3_is_routing_backed():
    w3 = _router_two_providers().web3(1)
    assert isinstance(w3.provider, RoutingProvider)


def test_eth_namespace_flows_through_routing_provider():
    """End-to-end: a real w3.eth.* call must flow through web3's request machinery into
    RoutingProvider.make_request and decode correctly (the whole point: consumer .eth.* unchanged)."""
    from web3 import Web3

    router = _router_two_providers()
    a, _ = router._providers[1]
    a.w3.provider.make_request = MagicMock(
        return_value={"jsonrpc": "2.0", "id": 1, "result": "0x10"}
    )

    w3 = Web3(RoutingProvider(router, 1))
    assert w3.eth.block_number == 16  # 0x10, decoded by web3 through the routing provider


def test_endpoint_uri_tracks_best_provider():
    router = _router_two_providers()
    rp = RoutingProvider(router, 1)
    assert rp.endpoint_uri == "http://a"  # priority 1
    router.report_failure(1)  # demote a
    assert rp.endpoint_uri == "http://b"  # now b


def _router_unreachable():
    router = RPCRouter()
    router.add_provider(1, ProviderConfig(name="a", url="http://127.0.0.1:1", priority=1))
    return router


def test_is_connected_false_when_provider_unreachable():
    assert RoutingProvider(_router_unreachable(), 1).is_connected() is False


def test_is_connected_raises_when_traceback_requested():
    with pytest.raises(ProviderConnectionError):
        RoutingProvider(_router_unreachable(), 1).is_connected(show_traceback=True)


def test_is_connected_false_without_providers():
    assert RoutingProvider(RPCRouter(), 1).is_connected() is False


def _http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.exceptions.HTTPError(f"{status} error", response=response)


@pytest.mark.parametrize("status", [400, 413])
def test_request_fault_raises_without_failover(status):
    """A 400 or 413 is the request's own fault: every provider would refuse it alike."""
    router = _router_two_providers()
    a, b = router._providers[1]
    a.w3.provider.make_request = MagicMock(side_effect=_http_error(status))
    b.w3.provider.make_request = MagicMock(
        return_value={"jsonrpc": "2.0", "id": 1, "result": "0x1"}
    )

    with pytest.raises(requests.exceptions.HTTPError):
        RoutingProvider(router, 1).make_request("eth_call", [])

    b.w3.provider.make_request.assert_not_called()
    assert a.cooldown_until == 0.0


@pytest.mark.parametrize("status", [401, 403, 429, 503])
def test_provider_refusal_fails_over(status):
    router = _router_two_providers()
    a, b = router._providers[1]
    a.w3.provider.make_request = MagicMock(side_effect=_http_error(status))
    b.w3.provider.make_request = MagicMock(
        return_value={"jsonrpc": "2.0", "id": 1, "result": "0x1"}
    )

    assert RoutingProvider(router, 1).make_request("eth_call", [])["result"] == "0x1"
    assert a.cooldown_until > 0
    assert b.cooldown_until == 0.0


def test_the_failed_provider_is_demoted_even_after_a_concurrent_report():
    """A concurrent request's report demotes A while this request is still waiting on A. This
    request's own report must land on A again, not on B, the selection by then."""
    router = _router_two_providers()
    a, b = router._providers[1]

    def concurrent_report_then_fail(method, params):
        router.report_failure(1)  # another request's report, made while A was still selected
        raise _http_error(429)

    a.w3.provider.make_request = MagicMock(side_effect=concurrent_report_then_fail)
    b.w3.provider.make_request = MagicMock(
        return_value={"jsonrpc": "2.0", "id": 1, "result": "0x1"}
    )

    assert RoutingProvider(router, 1).make_request("eth_call", [])["result"] == "0x1"
    assert b.cooldown_until == 0.0


def _router_one_provider():
    router = RPCRouter()
    router.add_provider(1, ProviderConfig(name="a", url="http://a", priority=1))
    return router


@pytest.fixture
def sleeps(monkeypatch):
    from web3_rpc_router import provider

    calls = []
    monkeypatch.setattr(provider.time, "sleep", calls.append)
    return calls


def test_the_only_provider_gets_a_second_attempt(sleeps):
    """One provider and nothing to fail over to: a transient failure is retried once on it,
    after a short pause, instead of reaching the caller."""
    from web3_rpc_router import provider

    router = _router_one_provider()
    (a,) = router._providers[1]
    a.w3.provider.make_request = MagicMock(
        side_effect=[
            requests.exceptions.ConnectionError("dropped"),
            {"jsonrpc": "2.0", "id": 1, "result": "0x1"},
        ]
    )

    assert RoutingProvider(router, 1).make_request("eth_call", [])["result"] == "0x1"
    assert a.w3.provider.make_request.call_count == 2
    assert sleeps == [provider._SAME_PROVIDER_RETRY_DELAY]


def test_the_only_provider_failing_twice_raises(sleeps):
    router = _router_one_provider()
    (a,) = router._providers[1]
    a.w3.provider.make_request = MagicMock(side_effect=_http_error(503))

    with pytest.raises(requests.exceptions.HTTPError):
        RoutingProvider(router, 1).make_request("eth_call", [])
    assert a.w3.provider.make_request.call_count == 2


def test_a_request_fault_on_the_only_provider_is_not_retried(sleeps):
    router = _router_one_provider()
    (a,) = router._providers[1]
    a.w3.provider.make_request = MagicMock(side_effect=_http_error(400))

    with pytest.raises(requests.exceptions.HTTPError):
        RoutingProvider(router, 1).make_request("eth_call", [])
    assert a.w3.provider.make_request.call_count == 1
    assert sleeps == []


def test_failing_over_to_another_provider_does_not_pause(sleeps):
    router = _router_two_providers()
    a, b = router._providers[1]
    a.w3.provider.make_request = MagicMock(side_effect=_http_error(429))
    b.w3.provider.make_request = MagicMock(
        return_value={"jsonrpc": "2.0", "id": 1, "result": "0x1"}
    )

    assert RoutingProvider(router, 1).make_request("eth_call", [])["result"] == "0x1"
    assert sleeps == []

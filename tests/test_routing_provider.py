from unittest.mock import MagicMock

import pytest
import requests

from web3_rpc_router import ProviderConfig, RPCRouter, RoutingProvider


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
    a.w3.provider.make_request = MagicMock(
        side_effect=requests.exceptions.ConnectionError("A")
    )
    b.w3.provider.make_request = MagicMock(
        side_effect=requests.exceptions.ConnectionError("B")
    )

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
    assert (
        w3.eth.block_number == 16
    )  # 0x10, decoded by web3 through the routing provider


def test_endpoint_uri_tracks_best_provider():
    router = _router_two_providers()
    rp = RoutingProvider(router, 1)
    assert rp.endpoint_uri == "http://a"  # priority 1
    router.report_failure(1)  # demote a
    assert rp.endpoint_uri == "http://b"  # now b

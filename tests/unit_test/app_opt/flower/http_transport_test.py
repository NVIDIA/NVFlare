# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import errno
import threading
from email.message import Message
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_opt.flower.connectors.http_client_connector import HttpClientConnector
from nvflare.app_opt.flower.connectors.http_server_connector import HttpServerConnector
from nvflare.app_opt.flower.http_transport import (
    HEARTBEAT_PATH,
    REQUEST_HEADERS,
    FleetHttpClient,
    HttpError,
    filter_headers,
    http_response,
    split_address,
    validate_request,
)


def request(path="/v1/fleet/register-node"):
    value = Shareable()
    value.update(
        method="POST",
        path=path,
        body=b"\x00\xffprotobuf",
        headers={
            "content-type": "application/protobuf",
            "flwr-public-key": "key",
            "flwr-signature": "signature",
            "flwr-timestamp": "timestamp",
            "flwr-package-version": "1.43.0",
        },
    )
    return value


@pytest.mark.parametrize(
    "path",
    [
        "http://evil/v1/fleet/get-run",
        "//evil/v1/fleet/get-run",
        "/v1/control/start-run",
        "/v1/runtime/pull-task",
        "/health",
        "/v1/fleet/../control/start-run",
        "/v1/fleet/get-run?x=1",
        "/v1/fleet/%67et-run",
    ],
)
def test_rejects_non_fleet_paths(path):
    with pytest.raises(HttpError) as error:
        validate_request(request(path), 1024)
    assert error.value.status == 404


@pytest.mark.parametrize(
    "key,value,status",
    [("method", "GET", 405), ("body", "text", 400), ("body", b"x" * 1025, 413), ("headers", {}, 415)],
)
def test_rejects_invalid_envelopes(key, value, status):
    req = request()
    req[key] = value
    with pytest.raises(HttpError) as error:
        validate_request(req, 1024)
    assert error.value.status == status


def test_preserves_only_required_headers_and_binary_content():
    req = request()
    req["headers"].update({"Host": "evil", "Authorization": "secret", "Connection": "keep-alive"})
    _, headers, body = validate_request(req, 1024)
    assert headers == request()["headers"]
    assert body == b"\x00\xffprotobuf"


@pytest.mark.parametrize("headers", [{"accept": "text/plain\r\nx: injected"}, {"Accept": "a", "accept": "b"}])
def test_rejects_ambiguous_or_injected_headers(headers):
    with pytest.raises(HttpError):
        filter_headers(headers, REQUEST_HEADERS)


@pytest.mark.parametrize("address", ["localhost:1234", "evil:80", "127.0.0.1:0", "127.0.0.1:65536"])
def test_only_loopback_targets(address):
    with pytest.raises(ValueError):
        split_address(address)


def context(site):
    ctx = MagicMock(spec=FLContext)
    ctx.get_peer_context.return_value = MagicMock(spec=FLContext)
    ctx.get_peer_context.return_value.get_identity_name.return_value = site
    return ctx


def test_server_revalidates_and_binds_flower_identity_to_site():
    connector = HttpServerConnector()
    connector.http_client = MagicMock()
    connector.http_client.send.return_value = http_response(200, b"registered")
    assert connector.send_request_to_flower(request(), context("site-1"))["body"] == b"registered"
    assert connector.send_request_to_flower(request("/v1/fleet/get-run"), context("site-2"))["status"] == 403
    assert connector.send_request_to_flower(request("/v1/control/start-run"), context("site-1"))["status"] == 404
    assert connector.http_client.send.call_count == 1
    assert connector.send_request_to_flower(request("/v1/fleet/get-run"), context("site-1"))["status"] == 200


def test_rejected_registration_does_not_claim_another_sites_key():
    connector = HttpServerConnector()
    connector.http_client = MagicMock()
    connector.http_client.send.side_effect = [http_response(401), http_response(200)]
    assert connector.send_request_to_flower(request(), context("bad-site"))["status"] == 401
    assert connector.send_request_to_flower(request(), context("site-1"))["status"] == 200


def test_request_path_does_not_poll_cli_status():
    connector = HttpServerConnector()
    connector.is_applet_stopped = MagicMock(side_effect=AssertionError("must not query CLI on Fleet request"))
    connector.send_request_to_flower = MagicMock(return_value=http_response(200))
    assert connector.process_app_request("request", request(), context("site-1"), Signal())["status"] == 200


def test_client_shutdown_cancels_forwarding_without_aborting_job():
    parent = Signal()
    connector = HttpClientConnector()
    connector.set_abort_signal(parent)
    connector.stop_applet = MagicMock()
    connector.stop(None)
    assert connector.abort_signal.triggered
    assert not parent.triggered
    assert connector._forward(request())["status"] == 503


@patch("nvflare.app_opt.flower.connectors.http_client_connector.FleetHttpServer")
@patch("nvflare.app_opt.flower.connectors.http_client_connector.get_local_addresses", return_value=["127.0.0.1:1234"])
def test_client_failed_start_closes_http_server(_addresses, server_type):
    connector = HttpClientConnector()
    connector.num_rounds = 1
    connector.set_abort_signal(Signal())
    connector.start_applet = MagicMock(side_effect=RuntimeError("cannot launch SuperNode"))
    connector.stop_applet = MagicMock()
    with pytest.raises(RuntimeError, match="cannot launch"):
        connector.start(context("site-1"))
    server_type.return_value.close.assert_called_once()
    assert connector.abort_signal.triggered


@patch("nvflare.app_opt.flower.connectors.http_server_connector.FleetHttpClient")
@patch("nvflare.app_opt.flower.connectors.http_server_connector.get_local_addresses", return_value=["127.0.0.1:1234"])
def test_server_failed_start_cleans_up_applet(_addresses, client_type):
    connector = HttpServerConnector()
    connector.start_applet = MagicMock(side_effect=RuntimeError("health timeout"))
    connector.stop_applet = MagicMock()
    with pytest.raises(RuntimeError, match="health timeout"):
        connector.start(context("site-1"))
    connector.stop_applet.assert_called_once()
    client_type.return_value.close.assert_called_once()


@pytest.mark.parametrize(
    "path,limits", [("/v1/fleet/pull-object", (2.0, 25.0)), ("/v1/fleet/send-node-heartbeat", (0.5, 3.5))]
)
def test_transaction_deadlines_fit_flower_http_deadlines(path, limits):
    connector = HttpClientConnector(per_msg_timeout=10.0, tx_timeout=100.0)
    connector.set_abort_signal(Signal())
    connector.send_request = MagicMock(return_value=http_response(200))
    assert connector._forward(request(path))["status"] == 200
    args = connector.send_request.call_args.kwargs
    assert (args["per_msg_timeout"], args["tx_timeout"]) == limits


@pytest.mark.parametrize("failure", [make_reply(ReturnCode.COMMUNICATION_ERROR), RuntimeError("transport unavailable")])
def test_uncertain_transaction_prevents_further_forwarding(failure):
    connector = HttpClientConnector()
    connector.set_abort_signal(Signal())
    connector.send_request = MagicMock(side_effect=[failure, http_response(200)])
    assert connector._forward(request())["status"] == 503
    assert connector._forward(request())["status"] == 503
    assert connector.send_request.call_count == 1
    stopped, code = connector._is_stopped()
    assert stopped and code != 0


def test_flower_controller_fails_job_on_client_transport_failure():
    from nvflare.app_common.tie.defs import Constant
    from nvflare.app_opt.flower.controller import FlowerController

    controller = FlowerController()
    controller._trigger_stop = MagicMock()
    controller._update_client_status = MagicMock()
    message = Shareable()
    message[Constant.MSG_KEY_EXIT_CODE] = Constant.EXIT_CODE_FAILED
    controller._process_client_done("done", message, context("site-1"))
    controller._trigger_stop.assert_called_once()


@pytest.mark.parametrize("failure", [make_reply(ReturnCode.COMMUNICATION_ERROR), RuntimeError("transport unavailable")])
def test_heartbeat_recovers_after_transient_failure(failure):
    connector = HttpClientConnector()
    connector.set_abort_signal(Signal())
    connector.send_request = MagicMock(side_effect=[failure, http_response(200)])
    assert connector._forward(request(HEARTBEAT_PATH))["status"] == 503
    assert not connector._server_done
    assert connector._exit_code == 0
    assert connector._forward(request(HEARTBEAT_PATH))["status"] == 200
    assert connector.send_request.call_count == 2


@pytest.mark.parametrize("path", [HEARTBEAT_PATH, "/v1/fleet/push-messages"])
@pytest.mark.parametrize("rc,done", [(ReturnCode.SERVICE_UNAVAILABLE, True), (ReturnCode.TASK_ABORTED, False)])
def test_clean_remote_shutdown_and_task_abort(path, rc, done):
    connector = HttpClientConnector()
    connector.set_abort_signal(Signal())
    connector.send_request = MagicMock(return_value=make_reply(rc))
    assert connector._forward(request(path))["status"] == 503
    assert connector._server_done == done
    assert connector._exit_code == 0


def test_timeout_warning_reports_effective_values(caplog):
    connector = HttpClientConnector(per_msg_timeout=10, tx_timeout=100)
    assert (connector.per_msg_timeout, connector.tx_timeout) == (2, 25)
    assert "per_msg_timeout=2.0, tx_timeout=25.0" in caplog.text


@pytest.mark.parametrize("error,status", [(TimeoutError(), 504), (ConnectionError(), 502)])
def test_upstream_failure_is_a_gateway_error_and_retains_identity(error, status):
    connector = HttpServerConnector()
    connector.http_client = MagicMock()
    connector.http_client.send.side_effect = error
    assert connector.send_request_to_flower(request(), context("site-1"))["status"] == status
    assert connector.send_request_to_flower(request(), context("site-2"))["status"] == 403


@pytest.mark.parametrize("length,body", [("garbage", b""), ("-1", b""), ("1025", b""), ("10", b"short")])
def test_invalid_upstream_content_length_is_502(length, body):
    client = FleetHttpClient("127.0.0.1:1234", max_message_bytes=1024)
    with (
        patch("nvflare.app_opt.flower.http_transport.socket.socket"),
        patch("nvflare.app_opt.flower.http_transport.http.client.HTTPConnection") as connection,
        patch.object(client, "_connect"),
    ):
        response = connection.return_value.getresponse.return_value
        response.getheader.return_value = length
        response.read.return_value = body
        with pytest.raises(HttpError) as error:
            client.send(request())
        assert error.value.status == 502
        connection.return_value.close.assert_called_once()
    assert not client._connections


@pytest.mark.parametrize("values", [["application/protobuf", "application/protobuf"], ["x" * 8193]])
def test_invalid_upstream_headers_are_502(values):
    client = FleetHttpClient("127.0.0.1:1234")
    with (
        patch("nvflare.app_opt.flower.http_transport.socket.socket"),
        patch("nvflare.app_opt.flower.http_transport.http.client.HTTPConnection") as connection,
        patch.object(client, "_connect"),
    ):
        response = connection.return_value.getresponse.return_value
        response.getheader.return_value = "0"
        response.read.return_value = b""
        response.status = 200
        response.headers = Message()
        for value in values:
            response.headers["content-type"] = value
        with pytest.raises(HttpError) as error:
            client.send(request())
        assert error.value.status == 502


def test_send_after_close_does_not_create_socket():
    client = FleetHttpClient("127.0.0.1:1234")
    client.close()
    with patch("nvflare.app_opt.flower.http_transport.socket.socket") as socket_type:
        assert client.send(request())["status"] == 503
    socket_type.assert_not_called()


def test_pending_connect_does_not_block_heartbeat_or_close():
    from tests.timing_utils import ThreadGroup

    client = FleetHttpClient("127.0.0.1:1234")
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    results = []

    def connect(sock, timeout):
        if timeout == 20:
            entered.set()
            release.wait()

    with (
        patch("nvflare.app_opt.flower.http_transport.socket.socket"),
        patch("nvflare.app_opt.flower.http_transport.http.client.HTTPConnection") as connection,
        patch.object(client, "_connect", side_effect=connect),
    ):
        connections = [MagicMock(), MagicMock()]
        connection.side_effect = connections
        response = connections[1].getresponse.return_value
        response.getheader.return_value = "0"
        response.read.return_value = b""
        response.headers = Message()
        response.status = 200
        with ThreadGroup(release) as workers:
            workers.start(lambda: results.append(client.send(request())))
            assert entered.wait(10)
            heartbeat_done = threading.Event()

            def heartbeat():
                assert client.send(request(HEARTBEAT_PATH))["status"] == 200
                heartbeat_done.set()

            workers.start(heartbeat)
            assert heartbeat_done.wait(5)
            workers.start(lambda: (client.close(), closed.set()))
            assert closed.wait(5)
        assert results[0]["status"] == 503
        assert not client._connections
        # Only the heartbeat was sent; the late connection cannot send or reopen.
        connections[0].request.assert_not_called()
        assert connections[1].request.call_count == 1
        assert all(conn.close.call_count == 1 for conn in connections)


def test_connect_wait_observes_cancellation():
    client = FleetHttpClient("127.0.0.1:1234")
    sock = MagicMock()
    sock.connect_ex.return_value = errno.EINPROGRESS
    with patch("nvflare.app_opt.flower.http_transport.selectors.DefaultSelector") as selector:
        selector.return_value.__enter__.return_value.select.side_effect = lambda _: client.close() or []
        with pytest.raises(OSError, match="cancelled"):
            client._connect(sock, 20)


def test_executor_logs_effective_timeouts():
    from nvflare.app_opt.flower.executor import FlowerExecutor

    executor = FlowerExecutor(per_msg_timeout=10, tx_timeout=100)
    executor.get_connector(None)
    executor.log_info = MagicMock()
    executor._log_states(None)
    logs = [call.args[1] for call in executor.log_info.call_args_list]
    assert "    self.per_msg_timeout=2.0" in logs
    assert "    self.tx_timeout=25.0" in logs

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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.f3.cellnet.connector_manager import ConnectorData
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.net_agent import NetAgent, SubnetMonitor, _Member
from nvflare.fuel.f3.message import Message


class _RecordingMonitor(SubnetMonitor):
    def __init__(self, subnet_id="subnet", members=None, threshold=10.0):
        super().__init__(subnet_id, members or ["site-1"], threshold)
        self.online = []
        self.offline = []

    def member_online(self, member_cell_fqcn: str):
        self.online.append(member_cell_fqcn)

    def member_offline(self, member_cell_fqcn: str):
        self.offline.append(member_cell_fqcn)


def _agent():
    cell = MagicMock()
    cell.get_fqcn.return_value = "site-1"
    cell.agents = {"site-2": object()}
    return NetAgent(cell), cell


def _reply(rc=ReturnCode.OK, payload=None, error=""):
    return Message(
        headers={MessageHeaderKey.RETURN_CODE: rc, MessageHeaderKey.ERROR: error},
        payload=payload,
    )


def test_init_registers_management_handlers():
    agent, cell = _agent()

    topics = {call.kwargs["topic"] for call in cell.register_request_cb.call_args_list}
    assert {"cells", "route", "peers", "speed", "heartbeat", "process_info"}.issubset(topics)
    assert agent.cell is cell


def test_subnet_monitor_tracks_state_transitions(monkeypatch):
    monitor = _RecordingMonitor(threshold=5.0)
    member = monitor.members["site-1"]
    monkeypatch.setattr("nvflare.fuel.f3.cellnet.net_agent.time.time", lambda: 100.0)

    monitor.put_member_online(member)
    monitor.put_member_online(member)
    assert member.state == _Member.STATE_ONLINE
    assert monitor.online == ["site-1"]

    member.last_heartbeat_time = 96.0
    monitor.put_member_offline(member)
    assert monitor.offline == []
    member.last_heartbeat_time = 90.0
    monitor.put_member_offline(member)
    assert member.state == _Member.STATE_OFFLINE
    assert monitor.offline == ["site-1"]


def test_subnet_monitor_delegates_stop_and_requires_agent():
    monitor = _RecordingMonitor()
    with pytest.raises(RuntimeError, match="No NetAgent"):
        monitor.stop_subnet()

    monitor.agent = MagicMock()
    monitor.agent.stop_subnet.return_value = {"site-1": "ok"}
    assert monitor.stop_subnet() == {"site-1": "ok"}


def test_add_monitor_validates_duplicates_and_starts_thread():
    agent, _ = _agent()
    monitor = _RecordingMonitor()

    with patch("nvflare.fuel.f3.cellnet.net_agent.threading.Thread") as thread_cls:
        agent.add_subnet_monitor(monitor)

    assert monitor.agent is agent
    assert agent.monitors == {"subnet": monitor}
    thread_cls.return_value.start.assert_called_once()
    with pytest.raises(ValueError, match="already exists"):
        agent.add_subnet_monitor(monitor)
    with pytest.raises(ValueError, match="monitor must be"):
        agent.add_subnet_monitor(object())

    agent.delete_subnet_monitor("subnet")
    assert agent.monitors == {}


def test_stop_subnet_only_broadcasts_to_online_members():
    agent, cell = _agent()
    monitor = _RecordingMonitor(members=["site-1", "site-2"])
    monitor.members["site-1"].state = _Member.STATE_ONLINE

    result = agent.stop_subnet(monitor)

    assert result is cell.broadcast_request.return_value
    assert cell.broadcast_request.call_args.kwargs["targets"] == ["site-1"]
    monitor.members["site-1"].state = _Member.STATE_OFFLINE
    assert agent.stop_subnet(monitor) is None


def test_close_is_idempotent_and_calls_callback():
    callback = MagicMock()
    cell = MagicMock()
    agent = NetAgent(cell, agent_closed_cb=callback)
    agent.heartbeat_thread = MagicMock()
    agent.heartbeat_thread.is_alive.return_value = True
    agent.monitor_thread = MagicMock()
    agent.monitor_thread.is_alive.return_value = True

    agent.close()
    agent.close()

    agent.heartbeat_thread.join.assert_called_once()
    agent.monitor_thread.join.assert_called_once()
    callback.assert_called_once()


def test_heartbeat_marks_known_member_online_and_ignores_unknown_subnets():
    agent, _ = _agent()
    monitor = _RecordingMonitor()
    agent.monitors = {"subnet": monitor}
    request = Message(
        headers={MessageHeaderKey.ORIGIN: "site-1"},
        payload={"subnet_id": "subnet"},
    )

    assert agent._do_heartbeat(request) is None
    assert monitor.online == ["site-1"]

    request.payload = {"subnet_id": "missing"}
    assert agent._do_heartbeat(request) is None
    request.payload = {"subnet_id": "subnet"}
    request.set_header(MessageHeaderKey.ORIGIN, "unknown")
    assert agent._do_heartbeat(request) is None


def test_simple_request_handlers():
    agent, cell = _agent()
    agent.stop = MagicMock()
    request = Message(headers={"header": "value"}, payload=b"payload")

    assert agent._do_route(request).payload == {"header": "value"}
    assert agent._do_peers(request).payload == ["site-2"]
    assert agent._do_echo(request).payload == b"payload"
    assert agent._do_stop_cell(request).payload is None
    agent.stop.assert_called_once()


@pytest.mark.parametrize(
    "reply, expected_error, expected_peers",
    [
        (_reply(payload=["site-1", "site-2"]), None, ["site-1", "site-2"]),
        (_reply(payload={"site-1": True}), "reply payload should be list", None),
        (_reply(ReturnCode.TIMEOUT), "return code: timeout", None),
    ],
)
def test_get_peers_validates_reply(reply, expected_error, expected_peers):
    agent, cell = _agent()
    cell.send_request.return_value = reply

    error, peers = agent.get_peers("site-2")

    assert peers == expected_peers
    if expected_error:
        assert expected_error in error["error"]
    else:
        assert error is None


def test_connector_inventory_and_url_use():
    agent, cell = _agent()
    listener = ConnectorData("listener", "tcp://listener:1", False, {"a": 1})
    connector = ConnectorData("connector", "tcp://connector:2", True, {"b": 2})
    cell.int_listener = listener
    cell.ext_listeners = {listener.connect_url: listener}
    cell.bb_ext_connector = connector
    cell.bb_int_connector = connector
    cell.adhoc_connectors = {"site-2": connector}

    inventory = agent._get_connectors()

    assert inventory["int_listener"]["type"] == "listener"
    assert inventory["bb_ext_connector"]["type"] == "connector"
    assert inventory["adhoc_connectors"]["site-2"]["url"] == connector.connect_url
    assert agent._get_url_use_of_cell(listener.connect_url) == "int_listen"
    assert agent._get_url_use_of_cell(connector.connect_url) == "bb_ext_connect"
    assert agent._get_url_use_of_cell("tcp://missing:3") == "none"
    assert agent._do_connectors(Message()).payload == inventory


@pytest.mark.parametrize(
    "reply, expected_error, expected",
    [
        (_reply(payload={"connector": 1}), {}, {"connector": 1}),
        (_reply(payload={}), {}, {}),
        (_reply(payload=[]), {"error": "reply payload should be dict but got <class 'list'>"}, {}),
        (_reply(ReturnCode.PROCESS_EXCEPTION), {"error": "processing error"}, {}),
    ],
)
def test_get_connectors_validates_reply(reply, expected_error, expected):
    agent, cell = _agent()
    cell.send_request.return_value = reply

    error, result = agent.get_connectors("site-2")

    assert result == expected
    assert error["error"] == expected_error["error"] if expected_error else error == {}


def test_route_operations_validate_payloads_and_return_codes():
    agent, cell = _agent()
    agent.get_route_info = MagicMock(return_value=({"reply": 1}, {"request": 2}))

    response = agent._do_start_route(Message(payload="site-2"))
    assert response.payload == {"request": {"request": 2}, "reply": {"reply": 1}}
    assert (
        agent._do_start_route(Message(payload="bad name")).get_header(MessageHeaderKey.RETURN_CODE)
        == ReturnCode.PROCESS_EXCEPTION
    )

    cell.send_request.return_value = _reply(payload={"reply": {"a": 1}, "request": {"b": 2}})
    assert agent.start_route("site-1", "site-2") == ("", {"a": 1}, {"b": 2})
    cell.send_request.return_value = _reply(ReturnCode.TIMEOUT)
    error, reply_headers, request_headers = agent.start_route("site-1", "site-2")
    assert error == "error in reply timeout"
    assert reply_headers == cell.send_request.return_value.headers
    assert request_headers == {}


def test_stress_and_bulk_tests_aggregate_remote_results():
    agent, cell = _agent()
    cell.broadcast_request.return_value = {
        "site-2": _reply(payload={"count": 2}),
        "site-3": _reply(ReturnCode.TIMEOUT),
    }

    assert agent.start_stress_test(["site-1"]) == {"error": "no targets for stress test"}
    assert agent.start_stress_test(["site-1", "site-2", "site-3"]) == {
        "site-2": {"count": 2},
        "site-3": "RC=timeout",
    }
    assert agent.start_bulk_test(["site-1"], 5) == {"error": "no targets for bulk test"}
    assert agent.start_bulk_test(["site-2", "site-3"], 5) == {
        "site-2": {"count": 2},
        "site-3": "RC=timeout",
    }


def test_stats_and_config_accessors_handle_success_and_failure():
    agent, cell = _agent()
    for method in (agent.get_msg_stats_table, agent.show_pool):
        cell.send_request.return_value = _reply(payload={"rows": [1]})
        args = ("site-2", "pool", "count") if method == agent.show_pool else ("site-2", "count")
        assert method(*args) == {"rows": [1]}
        cell.send_request.return_value = _reply(ReturnCode.TIMEOUT, error="late")
        expected = "timeout: late" if method == agent.show_pool else "error: timeout"
        assert method(*args) == expected

    for method in (agent.get_pool_list, agent.get_comm_config, agent.get_config_vars, agent.get_process_info):
        cell.send_request.return_value = _reply(payload={"value": 1})
        assert method("site-2") == {"value": 1}
        cell.send_request.return_value = _reply(ReturnCode.TIMEOUT, error="late")
        assert method("site-2") == "timeout: late"


def test_broadcast_to_subcells_filters_admin_clients_and_adjusts_timeout():
    agent, cell = _agent()
    admin = "_admin_123e4567-e89b-42d3-a456-426614174000"
    cell.get_sub_cell_names.return_value = (["site-1.job", admin], ["site-2"])
    cell.my_info = SimpleNamespace(is_root=False, is_on_server=False, gen=2)

    result = agent._broadcast_to_subs("topic", timeout=2.0)

    assert result is cell.broadcast_request.return_value
    assert cell.broadcast_request.call_args.kwargs["targets"] == ["site-1.job", "site-2"]
    assert cell.broadcast_request.call_args.kwargs["timeout"] == 1.0

    agent._broadcast_to_subs("topic", timeout=0.0)
    cell.fire_and_forget.assert_called_once()

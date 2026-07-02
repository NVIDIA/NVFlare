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
from unittest.mock import MagicMock, call, patch

import pytest

from nvflare.fuel.f3.cellnet.net_manager import NetManager, _to_int


def _manager():
    manager = NetManager.__new__(NetManager)
    manager.agent = MagicMock()
    manager.diagnose = True
    return manager


def _conn():
    conn = MagicMock()
    conn.get_prop.return_value = SimpleNamespace(usage="command usage")
    conn.append_table.return_value = MagicMock()
    return conn


def test_to_int_accepts_numbers_and_describes_errors():
    assert _to_int("12") == 12
    assert "not a valid number" in _to_int("bad")


def test_init_subscribes_stop_handler_and_get_spec_lists_commands():
    agent = MagicMock()
    with patch("nvflare.fuel.f3.cellnet.net_manager.DataBus") as data_bus_cls:
        manager = NetManager(agent, diagnose=True)

    data_bus_cls.return_value.subscribe.assert_called_once()
    spec = manager.get_spec()
    assert spec.name == "cellnet"
    assert len(spec.cmd_specs) == 17
    assert all(command.visible for command in spec.cmd_specs)


def test_stop_cellnet_stops_agent_and_reports():
    manager = _manager()
    conn = _conn()

    manager._stop_cellnet("topic", conn, MagicMock())

    manager.agent.stop.assert_called_once()
    conn.append_string.assert_called_once_with("Cellnet Stopped")


def test_cells_reports_errors_valid_cells_and_total():
    manager = _manager()
    conn = _conn()
    manager.agent.request_cells_info.return_value = ("partial failure", ["server", "site-1", "bad name"])

    manager._cmd_cells(conn, ["cells"])

    conn.append_error.assert_called_once_with("partial failure")
    assert conn.append_string.call_args_list[-1].args == ("Total Cells: 2",)


def test_url_use_validates_args_and_removes_unused_cells():
    manager = _manager()
    conn = _conn()

    manager._cmd_url_use(conn, ["url_use"])
    conn.append_string.assert_called_with("Usage: command usage")

    manager.agent.get_url_use.return_value = {"server": "none", "site-1": "bb_ext_connect"}
    manager._cmd_url_use(conn, ["url_use", "tcp://server:8002"])
    conn.append_dict.assert_called_once_with({"site-1": "bb_ext_connect"})

    manager.agent.get_url_use.return_value = {"server": "none"}
    manager._cmd_url_use(conn, ["url_use", "unused"])
    conn.append_string.assert_called_with("No cell uses unused")


def test_route_reports_request_reply_and_errors():
    manager = _manager()
    conn = _conn()
    manager.agent.start_route.return_value = ("route failed", {"reply": 1}, {"request": 2})

    manager._cmd_route(conn, ["route"])
    manager._cmd_route(conn, ["route", "site-2", "site-1"])

    manager.agent.start_route.assert_called_once_with("site-1", "site-2")
    conn.append_error.assert_called_once_with("route failed")
    assert conn.append_dict.call_args_list[0].args == ({"request": 2},)
    assert conn.append_dict.call_args_list[1].args == ({"reply": 1},)


def test_peers_and_connectors_handle_empty_errors_and_results():
    manager = _manager()
    conn = _conn()

    manager._cmd_peers(conn, ["peers"])
    manager.agent.get_peers.return_value = ({"error": "late"}, ["site-1", "site-2"])
    manager._cmd_peers(conn, ["peers", "server"])
    assert conn.append_string.call_args_list[-1].args == ("Total Agents: 2",)

    manager.agent.get_peers.return_value = (None, [])
    manager._cmd_peers(conn, ["peers", "server"])
    conn.append_string.assert_called_with("No peers")

    manager._cmd_connectors(conn, ["conns"])
    manager.agent.get_connectors.return_value = ({"error": "late"}, {"listener": 1})
    manager._cmd_connectors(conn, ["conns", "server"])
    assert conn.append_dict.call_args_list[-2:] == [call({"error": "late"}), call({"listener": 1})]


def test_speed_test_validates_numbers_and_delegates():
    manager = _manager()
    conn = _conn()
    manager.agent.speed_test.return_value = {"average": 0.1}

    manager._cmd_speed_test(conn, ["speed"])
    manager._cmd_speed_test(conn, ["speed", "site-1", "site-2", "bad"])
    manager._cmd_speed_test(conn, ["speed", "site-1", "site-2", "2", "bad"])
    manager._cmd_speed_test(conn, ["speed", "site-1", "site-2", "2", "4"])

    manager.agent.speed_test.assert_called_once_with(from_fqcn="site-1", to_fqcn="site-2", num_tries=2, payload_size=4)
    conn.append_dict.assert_called_once_with({"average": 0.1})
    assert conn.append_error.call_count == 2


def test_stress_test_aggregates_errors_and_uses_defaults():
    manager = _manager()
    conn = _conn()
    manager.agent.request_cells_info.return_value = ("partial", ["site-1", "site-2"])
    manager.agent.start_stress_test.return_value = {
        "site-1": {"errors": {"site-2": 0}, "counts": {"site-2": 2}},
        "site-2": {"errors": {"site-1": 3}},
        "server": "timeout",
    }

    manager._cmd_stress_test(conn, ["stress", "20", "8"])

    manager.agent.start_stress_test.assert_called_once_with(targets=["site-1", "site-2"], num_rounds=20, timeout=8)
    assert manager.agent.start_stress_test.return_value["site-1"].get("errors") is None
    conn.append_string.assert_called_with("total errors: 3")
    conn.append_error.assert_called_once_with("partial")


@pytest.mark.parametrize("args", [["stress", "bad"], ["stress", "2", "bad"]])
def test_stress_test_rejects_invalid_numbers(args):
    manager = _manager()
    conn = _conn()
    manager._cmd_stress_test(conn, args)
    conn.append_error.assert_called_once()
    manager.agent.start_stress_test.assert_not_called()


def test_bulk_test_validates_size_and_delegates():
    manager = _manager()
    conn = _conn()
    manager.agent.request_cells_info.return_value = ("partial", ["site-1"])
    manager.agent.start_bulk_test.return_value = {"site-1": "queued"}

    manager._cmd_bulk_test(conn, ["bulk", "3"])

    manager.agent.start_bulk_test.assert_called_once_with(["site-1"], 3)
    conn.append_error.assert_called_once_with("partial")
    conn.append_dict.assert_called_once_with({"site-1": "queued"})

    invalid = _conn()
    manager._cmd_bulk_test(invalid, ["bulk", "bad"])
    invalid.append_error.assert_called_once()


@pytest.mark.parametrize(
    "method_name, agent_method, args",
    [
        ("_cmd_msg_stats", "get_msg_stats_table", ["msg_stats", "server", "avg"]),
        ("_cmd_show_pool", "show_pool", ["show_pool", "server", "latency", "max"]),
        ("_cmd_list_pools", "get_pool_list", ["list_pools", "server"]),
        ("_cmd_process_info", "get_process_info", ["process_info", "server"]),
    ],
)
def test_table_commands_render_rows(method_name, agent_method, args):
    manager = _manager()
    conn = _conn()
    reply = {"headers": ["name", "value"], "rows": [["one", 1], ["two", 2]]}
    getattr(manager.agent, agent_method).return_value = reply

    getattr(manager, method_name)(conn, args)

    conn.append_table.assert_called_once_with(reply["headers"])
    assert conn.append_table.return_value.add_row.call_count == 2


@pytest.mark.parametrize(
    "method_name, agent_method, args",
    [
        ("_cmd_msg_stats", "get_msg_stats_table", ["msg_stats"]),
        ("_cmd_show_pool", "show_pool", ["show_pool", "server"]),
        ("_cmd_list_pools", "get_pool_list", ["list_pools"]),
        ("_cmd_process_info", "get_process_info", ["process_info"]),
    ],
)
def test_table_commands_show_usage(method_name, agent_method, args):
    manager = _manager()
    conn = _conn()
    getattr(manager, method_name)(conn, args)
    conn.append_string.assert_called_once_with("Usage: command usage")
    getattr(manager.agent, agent_method).assert_not_called()


@pytest.mark.parametrize(
    "method_name, agent_method, args",
    [
        ("_cmd_msg_stats", "get_msg_stats_table", ["msg_stats", "server", "bad"]),
        ("_cmd_show_pool", "show_pool", ["show_pool", "server", "pool", "bad"]),
    ],
)
def test_histogram_commands_reject_invalid_modes(method_name, agent_method, args):
    manager = _manager()
    conn = _conn()
    getattr(manager, method_name)(conn, args)
    conn.append_error.assert_called_once()
    getattr(manager.agent, agent_method).assert_not_called()


@pytest.mark.parametrize("reply", ["remote error", ["wrong type"]])
def test_table_commands_validate_remote_reply(reply):
    manager = _manager()
    conn = _conn()
    manager.agent.get_pool_list.return_value = reply
    manager._cmd_list_pools(conn, ["list_pools", "server"])
    conn.append_error.assert_called_once()


@pytest.mark.parametrize(
    "method_name, agent_method",
    [
        ("_cmd_show_comm_config", "get_comm_config"),
        ("_cmd_show_config_vars", "get_config_vars"),
    ],
)
def test_dict_commands_validate_and_render_reply(method_name, agent_method):
    manager = _manager()
    conn = _conn()
    method = getattr(manager, method_name)

    method(conn, ["command"])
    conn.append_string.assert_called_with("Usage: command usage")

    getattr(manager.agent, agent_method).return_value = "remote error"
    method(conn, ["command", "server"])
    conn.append_error.assert_called_with("remote error")

    getattr(manager.agent, agent_method).return_value = []
    method(conn, ["command", "server"])
    assert "expect dict" in conn.append_error.call_args.args[0]

    getattr(manager.agent, agent_method).return_value = {"key": "value"}
    method(conn, ["command", "server"])
    conn.append_dict.assert_called_once_with({"key": "value"})


def test_change_root_and_stop_commands():
    manager = _manager()
    conn = _conn()

    manager._cmd_change_root(conn, ["change_root"])
    manager._cmd_change_root(conn, ["change_root", "tcp://new-root:8002"])
    manager.agent.change_root.assert_called_once_with("tcp://new-root:8002")

    manager._cmd_stop_cell(conn, ["stop_cell"])
    manager.agent.stop_cell.return_value = "ok"
    manager._cmd_stop_cell(conn, ["stop_cell", "site-1"])
    conn.append_string.assert_called_with("Asked site-1 to stop: ok")

    manager._cmd_stop_net(conn, ["stop_net"])
    manager.agent.stop.assert_called_once()
    conn.append_shutdown.assert_called_once_with("Cellnet Stopped")

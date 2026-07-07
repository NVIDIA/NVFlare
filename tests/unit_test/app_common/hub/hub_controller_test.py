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

import json
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, OperatorConfigKey, OperatorMethod, Task, TaskOperatorKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.hub.hub_controller import BcastOperator, HubController, RelayOperator
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.pipe import Message, Pipe, Topic
from tests.unit_test.fl_context_helper import make_fl_context


class _Aggregator(Aggregator):
    def __init__(self):
        super().__init__()
        self.accepted = []
        self.reset_count = 0

    def reset(self, fl_ctx):
        self.reset_count += 1

    def accept(self, shareable, fl_ctx):
        self.accepted.append(shareable)
        return True

    def aggregate(self, fl_ctx):
        return Shareable({"count": len(self.accepted)})


class _Generator(ShareableGenerator):
    def learnable_to_shareable(self, model, fl_ctx):
        return Shareable({"model": model})

    def shareable_to_learnable(self, shareable, fl_ctx):
        return Learnable({"source": shareable})


class _Persistor(LearnablePersistor):
    def load(self, fl_ctx):
        return Learnable({"weights": 1})

    def save(self, learnable, fl_ctx):
        return None


class _Pipe(Pipe):
    def __init__(self):
        super().__init__(Mode.ACTIVE)
        self.opened = []

    def open(self, name):
        self.opened.append(name)

    def clear(self):
        return None

    def send(self, msg, timeout=None):
        return True

    def receive(self, timeout=None):
        return None

    def close(self):
        return None

    def can_resend(self):
        return False

    def export(self, export_mode):
        return {}


def _context(engine=None):
    return make_fl_context(engine=engine, identity_name="server", run_num="job-1")


def _hub():
    hub = HubController("pipe")
    hub.log_debug = MagicMock()
    hub.log_info = MagicMock()
    hub.log_warning = MagicMock()
    hub.log_error = MagicMock()
    hub.log_exception = MagicMock()
    hub.fire_event = MagicMock()
    hub.operator_descs = {}
    return hub


def test_bcast_operator_validates_aggregator_and_operates():
    operator = BcastOperator()
    engine = MagicMock()
    fl_ctx = _context(engine)

    with pytest.raises(RuntimeError, match="missing aggregator"):
        operator._get_aggregator({}, fl_ctx)
    engine.get_component.return_value = None
    with pytest.raises(RuntimeError, match="no aggregator"):
        operator._get_aggregator({TaskOperatorKey.AGGREGATOR: "aggr"}, fl_ctx)
    engine.get_component.return_value = object()
    with pytest.raises(RuntimeError, match="must be Aggregator"):
        operator._get_aggregator({TaskOperatorKey.AGGREGATOR: "aggr"}, fl_ctx)

    aggregator = _Aggregator()
    engine.get_component.return_value = aggregator
    engine.get_clients.return_value = [Client("site-1", "token")]
    controller = MagicMock()
    result = operator.operate(
        {
            TaskOperatorKey.AGGREGATOR: "aggr",
            TaskOperatorKey.MIN_TARGETS: 5,
            TaskOperatorKey.TARGETS: ["site-1"],
            TaskOperatorKey.TIMEOUT: 3,
        },
        controller,
        "train",
        Shareable(),
        Signal(),
        fl_ctx,
    )
    assert result["count"] == 0
    assert aggregator.reset_count == 1
    assert controller.broadcast_and_wait.call_args.kwargs["min_responses"] == 1
    assert operator.current_aggregator is None


def test_bcast_operator_processes_results_and_late_results():
    operator = BcastOperator()
    aggregator = _Aggregator()
    task = Task("train", Shareable(), props={operator._PROP_AGGR: aggregator})
    client_task = ClientTask(Client("site-1", "token"), task)
    client_task.result = Shareable({"value": 1})

    operator._process_bcast_result(client_task, FLContext())
    assert client_task.result is None
    assert len(aggregator.accepted) == 1

    operator.current_aggregator = aggregator
    operator.process_result_of_unknown_task(
        Client("site-1", "token"), "train", "task", Shareable({"late": True}), FLContext()
    )
    assert len(aggregator.accepted) == 2


def test_relay_operator_validates_optional_components():
    operator = RelayOperator()
    engine = MagicMock()
    fl_ctx = _context(engine)
    assert operator._get_shareable_generator({}, fl_ctx) is None
    assert operator._get_persistor({}, fl_ctx) is None

    engine.get_component.return_value = None
    with pytest.raises(RuntimeError, match="no shareable generator"):
        operator._get_shareable_generator({TaskOperatorKey.SHAREABLE_GENERATOR: "gen"}, fl_ctx)
    with pytest.raises(RuntimeError, match="no persistor"):
        operator._get_persistor({TaskOperatorKey.PERSISTOR: "persistor"}, fl_ctx)

    engine.get_component.return_value = object()
    with pytest.raises(RuntimeError, match="must be ShareableGenerator"):
        operator._get_shareable_generator({TaskOperatorKey.SHAREABLE_GENERATOR: "gen"}, fl_ctx)
    with pytest.raises(RuntimeError, match="must be LearnablePersistor"):
        operator._get_persistor({TaskOperatorKey.PERSISTOR: "persistor"}, fl_ctx)


def test_relay_operator_operates_and_processes_result():
    operator = RelayOperator()
    generator = _Generator()
    persistor = _Persistor()
    engine = MagicMock()
    engine.get_component.side_effect = lambda comp_id: generator if comp_id == "gen" else persistor
    fl_ctx = _context(engine)
    controller = MagicMock()
    task_data = Shareable()
    task_data.set_header(AppConstants.CURRENT_ROUND, 2)

    def relay(task, **kwargs):
        client_task = ClientTask(Client("site-1", "token"), task)
        client_task.result = Shareable({"update": 1})
        operator._process_relay_result(client_task, fl_ctx)

    controller.relay_and_wait.side_effect = relay
    result = operator.operate(
        {
            TaskOperatorKey.SHAREABLE_GENERATOR: "gen",
            TaskOperatorKey.PERSISTOR: "persistor",
            TaskOperatorKey.TARGETS: ["site-1"],
            TaskOperatorKey.TASK_ASSIGNMENT_TIMEOUT: 3,
        },
        controller,
        "train",
        task_data,
        Signal(),
        fl_ctx,
    )
    assert result["update"] == 1
    assert fl_ctx.get_prop(AppConstants.GLOBAL_MODEL) == {"weights": 1}

    abort_signal = Signal()
    abort_signal.trigger(True)
    assert operator.operate({}, controller, "train", Shareable(), abort_signal, fl_ctx) is None


def test_start_and_event_handlers_load_config_and_pipe(tmp_path):
    app_config = tmp_path / "config.json"
    app_config.write_text(json.dumps({OperatorConfigKey.OPERATORS: {"train": {TaskOperatorKey.METHOD: "bcast"}}}))
    engine = MagicMock()
    engine.get_workspace.return_value.get_server_app_config_file_path.return_value = str(app_config)
    pipe = _Pipe()
    engine.get_component.return_value = pipe
    hub = _hub()
    fl_ctx = _context(engine)

    hub.start_controller(fl_ctx)
    assert hub.project_name == "server"
    assert hub.operator_descs["train"][TaskOperatorKey.METHOD] == "bcast"

    with patch("nvflare.app_common.hub.hub_controller.PipeHandler") as handler_cls:
        hub.handle_event(EventType.START_RUN, fl_ctx)
    assert pipe.opened
    assert hub.pipe_handler is handler_cls.return_value
    hub.handle_event(EventType.END_RUN, fl_ctx)
    assert hub.run_ended


def test_get_operator_and_resolve_description():
    engine = MagicMock()
    fl_ctx = _context(engine)
    hub = _hub()
    assert "missing method" in hub._get_operator("train", {}, fl_ctx)[1]

    engine.get_component.return_value = None
    operator, error = hub._get_operator("train", {TaskOperatorKey.METHOD: OperatorMethod.BROADCAST}, fl_ctx)
    assert isinstance(operator, BcastOperator)
    assert error == ""

    assert "no operator" in hub._get_operator("train", {TaskOperatorKey.METHOD: "missing"}, fl_ctx)[1]
    engine.get_component.return_value = object()
    assert "must be OperatorSpec" in hub._get_operator("train", {TaskOperatorKey.METHOD: "custom"}, fl_ctx)[1]

    hub.project_name = "project"
    hub.operator_descs = {
        "train": {"general": True},
        "project.train": {"project": True},
    }
    desc = {TaskOperatorKey.OP_ID: "train", "original": True}
    hub._resolve_op_desc(desc, fl_ctx)
    assert desc["project"] is True

    hub.operator_descs.pop("project.train")
    desc = {TaskOperatorKey.OP_ID: "train"}
    hub._resolve_op_desc(desc, fl_ctx)
    assert desc["general"] is True


def test_control_flow_starts_stops_and_aborts_on_exception():
    hub = _hub()
    hub.pipe_handler = MagicMock()
    hub._control_flow = MagicMock()
    signal = Signal()
    hub.control_flow(signal, FLContext())
    hub.pipe_handler.start.assert_called_once()
    hub.pipe_handler.stop.assert_called_once()

    hub._control_flow.side_effect = RuntimeError("failed")
    hub._abort = MagicMock()
    hub.control_flow(signal, FLContext())
    hub._abort.assert_called_once()


@pytest.mark.parametrize("stop_kind", ["run_ended", "abort_signal"])
def test_control_loop_stops_when_run_or_signal_ends(stop_kind):
    hub = _hub()
    hub.pipe_handler = MagicMock()
    signal = Signal()
    if stop_kind == "run_ended":
        hub.run_ended = True
    else:
        signal.trigger(True)
    hub._control_flow(signal, FLContext())
    hub.pipe_handler.notify_abort.assert_called_once_with("")


@pytest.mark.parametrize(
    "message, error",
    [
        (Message.new_request("task", object()), "must be Shareable"),
        (Message.new_request("task", Shareable()), "missing task name"),
    ],
)
def test_control_loop_aborts_invalid_messages(message, error):
    hub = _hub()
    hub.pipe_handler = MagicMock()
    hub.pipe_handler.get_next.return_value = message
    signal = Signal()
    hub._control_flow(signal, FLContext())
    assert signal.triggered
    assert error in hub.pipe_handler.notify_abort.call_args.args[0]


def test_control_loop_handles_peer_status_and_non_request():
    hub = _hub()
    hub.pipe_handler = MagicMock()
    hub.pipe_handler.get_next.return_value = Message.new_request(Topic.END, Shareable())
    hub._control_flow(Signal(), FLContext())

    hub.pipe_handler.get_next.side_effect = [
        Message.new_reply("ignored", Shareable(), "request"),
        Message.new_request(Topic.END, Shareable()),
    ]
    with patch("nvflare.app_common.hub.hub_controller.time.sleep"):
        hub._control_flow(Signal(), FLContext())
    hub.log_info.assert_called()


def test_control_loop_invokes_operator_and_replies():
    engine = MagicMock()
    fl_ctx = _context(engine)
    hub = _hub()
    operator = MagicMock(spec=BcastOperator)
    operator.operate.return_value = Shareable({"result": 1})
    engine.get_component.return_value = operator
    task_data = Shareable()
    task_data.set_header(ReservedHeaderKey.TASK_NAME, "train")
    task_data.set_header(ReservedHeaderKey.TASK_ID, "task-1")
    task_data.set_header(ReservedHeaderKey.TASK_OPERATOR, {TaskOperatorKey.METHOD: "custom"})
    hub.pipe_handler = MagicMock()
    hub.pipe_handler.get_next.side_effect = [
        Message.new_request("task", task_data),
        Message.new_request(Topic.END, Shareable()),
    ]

    with patch("nvflare.app_common.hub.hub_controller.time.sleep"):
        hub._control_flow(Signal(), fl_ctx)

    operator.operate.assert_called_once()
    reply = hub.pipe_handler.send_to_peer.call_args.args[0]
    assert reply.data["result"] == 1
    assert hub.current_operator is None


def test_late_results_are_forwarded_only_to_current_operator():
    hub = _hub()
    operator = MagicMock()
    hub.current_task_name = "train"
    hub.current_operator = operator
    client = Client("site-1", "token")
    result = Shareable()
    hub.process_result_of_unknown_task(client, "train", "task", result, FLContext())
    operator.process_result_of_unknown_task.assert_called_once()

    hub.process_result_of_unknown_task(client, "other", "task", result, FLContext())
    hub.log_warning.assert_called_once()
    assert hub.stop_controller(FLContext()) is None

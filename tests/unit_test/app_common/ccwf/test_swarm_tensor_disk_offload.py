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

import os
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedTopic
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.ccwf.client_ctl import ClientSideController
from nvflare.app_common.ccwf.swarm_client_ctl import SwarmClientController
from nvflare.private.defs import CellChannel


class _MockCell:
    def __init__(self):
        self.ctx = {"enable_tensor_disk_offload": False}
        self.decode_pass_through_relay_topics = set()

    def get_fobs_context(self):
        return dict(self.ctx)

    def update_fobs_context(self, props: dict):
        self.ctx.update(props)


class _MockEngine:
    def __init__(self, cell, aggregator):
        self.cell = cell
        self.aggregator = aggregator

    def get_cell(self):
        return self.cell

    def get_component(self, component_id):
        return self.aggregator


class _FakeThread:
    def __init__(self, name, stop_on_join, on_join=None):
        self.name = name
        self.stop_on_join = stop_on_join
        self.on_join = on_join
        self.join_calls = []
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.join_calls.append(timeout)
        if self.on_join:
            self.on_join()
        if self.stop_on_join:
            self.alive = False


def test_swarm_controller_owns_offload_root_without_enabling_cell_globally():
    cell = _MockCell()
    controller = SwarmClientController(enable_tensor_disk_offload=True)
    controller.engine = _MockEngine(cell, InTimeAccumulateWeightedAggregator())
    controller.log_debug = MagicMock()
    fl_ctx = MagicMock()
    fl_ctx.get_job_id.return_value = "swarm-job"
    fl_ctx.get_prop.return_value = False

    with (
        patch.object(ClientSideController, "start_run", autospec=True),
        patch.object(ClientSideController, "finalize", autospec=True) as super_finalize,
        patch("nvflare.app_common.ccwf.swarm_client_ctl.threading.Thread") as thread_cls,
    ):
        thread_cls.return_value.is_alive.return_value = False
        controller.start_run(fl_ctx)

        offload_root = controller._tensor_disk_offload_root_dir
        assert cell.ctx["enable_tensor_disk_offload"] is False
        assert "tensor_disk_offload_root_dir" not in cell.ctx
        assert os.path.isdir(offload_root)
        thread_cls.return_value.start.assert_called_once_with()

        controller.finalize(fl_ctx)
        assert os.path.isdir(offload_root)

        controller.workflow_done = True
        controller.handle_event(EventType.END_RUN, fl_ctx)

    assert cell.ctx["enable_tensor_disk_offload"] is False
    assert "tensor_disk_offload_root_dir" not in cell.ctx
    assert not os.path.exists(offload_root)
    assert controller._tensor_disk_offload_root_dir is None
    super_finalize.assert_called_once_with(controller, fl_ctx)


def test_swarm_controller_rejects_non_boolean_tensor_disk_offload():
    with pytest.raises(TypeError, match="enable_tensor_disk_offload"):
        SwarmClientController(enable_tensor_disk_offload="yes")


def test_finalize_preserves_offload_root_until_end_run():
    cell = _MockCell()
    controller = SwarmClientController(enable_tensor_disk_offload=True)
    controller.engine = _MockEngine(cell, InTimeAccumulateWeightedAggregator())
    controller.log_debug = MagicMock()
    fl_ctx = MagicMock()
    fl_ctx.get_job_id.return_value = "swarm-job"
    fl_ctx.get_prop.return_value = False

    with (
        patch.object(ClientSideController, "start_run", autospec=True),
        patch.object(ClientSideController, "finalize", autospec=True),
        patch("nvflare.app_common.ccwf.swarm_client_ctl.threading.Thread") as thread_cls,
    ):
        thread_cls.return_value.is_alive.return_value = False
        controller.start_run(fl_ctx)
        root_dir = controller._tensor_disk_offload_root_dir
        temp_dir = os.path.join(root_dir, "nvflare_tensors")
        os.mkdir(temp_dir)

        controller.finalize(fl_ctx)
        assert os.path.isdir(root_dir)

        controller.workflow_done = True
        controller.handle_event(EventType.END_RUN, fl_ctx)
        assert not os.path.exists(root_dir)
        assert controller._tensor_disk_offload_root_dir is None


def test_secure_swarm_relays_forwarded_learn_tensors_through_client_job():
    cell = _MockCell()
    controller = SwarmClientController(enable_tensor_disk_offload=True)
    controller.engine = _MockEngine(cell, InTimeAccumulateWeightedAggregator())
    controller.log_debug = MagicMock()
    fl_ctx = MagicMock()
    fl_ctx.get_job_id.return_value = "swarm-job"
    fl_ctx.get_prop.side_effect = lambda key, default=None: key == FLContextKey.SECURE_MODE
    route = (CellChannel.AUX_COMMUNICATION, ReservedTopic.DO_TASK)

    with (
        patch.object(ClientSideController, "start_run", autospec=True),
        patch.object(ClientSideController, "finalize", autospec=True),
        patch("nvflare.app_common.ccwf.swarm_client_ctl.threading.Thread") as thread_cls,
    ):
        thread_cls.return_value.is_alive.return_value = False
        controller.start_run(fl_ctx)
        assert route in cell.decode_pass_through_relay_topics
        controller.finalize(fl_ctx)
        assert route in cell.decode_pass_through_relay_topics

        controller.workflow_done = True
        controller.handle_event(EventType.END_RUN, fl_ctx)

    assert route not in cell.decode_pass_through_relay_topics


def test_cleanup_waits_for_controller_owned_threads_before_removing_root(tmp_path):
    root_dir = tmp_path / "offload"
    root_dir.mkdir()
    controller = SwarmClientController(enable_tensor_disk_offload=True, learn_task_abort_timeout=1.0)
    controller._tensor_disk_offload_root_dir = str(root_dir)
    controller.log_warning = MagicMock()

    root_exists_during_join = []

    def record_root_state():
        root_exists_during_join.append(root_dir.exists())

    controller.learn_thread = _FakeThread("learn", stop_on_join=True, on_join=record_root_state)
    controller._aggr_thread = _FakeThread("aggregate", stop_on_join=True, on_join=record_root_state)

    controller._cleanup_tensor_disk_offload(MagicMock())

    assert len(controller.learn_thread.join_calls) == 1
    assert len(controller._aggr_thread.join_calls) == 1
    assert 0.0 < controller.learn_thread.join_calls[0] <= 1.0
    assert 0.0 < controller._aggr_thread.join_calls[0] <= 1.0
    assert root_exists_during_join == [True, True]
    assert not root_dir.exists()
    assert controller._tensor_disk_offload_root_dir is None
    controller.log_warning.assert_not_called()


def test_cleanup_uses_shared_deadline_then_removes_root_after_timeout(tmp_path):
    root_dir = tmp_path / "offload"
    root_dir.mkdir()
    controller = SwarmClientController(enable_tensor_disk_offload=True, learn_task_abort_timeout=1.0)
    controller._tensor_disk_offload_root_dir = str(root_dir)
    controller.log_warning = MagicMock()
    controller.learn_thread = _FakeThread("learn", stop_on_join=False)
    controller._aggr_thread = _FakeThread("aggregate", stop_on_join=False)

    with patch("nvflare.app_common.ccwf.swarm_client_ctl.time.monotonic", side_effect=[10.0, 10.25, 10.75]):
        controller._cleanup_tensor_disk_offload(MagicMock())

    assert controller.learn_thread.join_calls == [pytest.approx(0.75)]
    assert controller._aggr_thread.join_calls == [pytest.approx(0.25)]
    assert not root_dir.exists()
    assert controller._tensor_disk_offload_root_dir is None
    warning = controller.log_warning.call_args.args[1]
    assert "learn" in warning
    assert "aggregate" in warning
    assert str(root_dir) in warning

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Tests for receiver-side PASS_THROUGH handling on SwarmClientController.

Covers:
  1. _has_lazy_refs() correctly detects LazyDownloadRef in nested structures.
  2. Swarm resolves learner input unless ClientAPIExecutor uses external_process.
  3. _scatter() preserves its local copy and requests PASS_THROUGH for remote tasks.
"""
import unittest
from unittest.mock import MagicMock

import numpy as np

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.shareable import ReservedHeaderKey
from nvflare.app_common.ccwf.swarm_client_ctl import SwarmClientController
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor, ExecutionMode
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef


def _make_shareable_with_lazy_refs():
    lazy_data = {
        "layer.weight": LazyDownloadRef(fqcn="aggr-cj.subprocess", ref_id="ref-x", item_id="T0"),
        "layer.bias": LazyDownloadRef(fqcn="aggr-cj.subprocess", ref_id="ref-x", item_id="T1"),
    }
    dxo = DXO(data_kind=DataKind.WEIGHTS, data=lazy_data)
    return dxo.to_shareable()


def _make_shareable_with_real_arrays():
    real_data = {
        "layer.weight": np.zeros((4, 4), dtype=np.float32),
        "layer.bias": np.zeros((4,), dtype=np.float32),
    }
    dxo = DXO(data_kind=DataKind.WEIGHTS, data=real_data)
    return dxo.to_shareable()


def _make_controller():
    """Build a minimal SwarmClientController for testing."""
    ctl = SwarmClientController.__new__(SwarmClientController)
    ctl.logger = MagicMock()
    ctl.log_info = MagicMock()
    ctl.log_error = MagicMock()
    ctl.log_debug = MagicMock()
    ctl.log_warning = MagicMock()
    ctl.me = "site-1"
    ctl.enable_tensor_disk_offload = False
    ctl.metric_comparator = None
    ctl.metric_comparator_id = None
    ctl.report_learn_result_task_name = "swarm_report_learn_result"
    ctl.request_to_submit_learn_result_task_name = "swarm_request_to_submit_learn_result"
    ctl.max_concurrent_submissions = 1
    ctl.request_to_submit_result_max_wait = None
    ctl.request_to_submit_result_msg_timeout = 5.0
    ctl.request_to_submit_result_interval = 0.0
    ctl.learn_task_timeout = None
    ctl.min_responses_required = 1
    ctl.wait_time_after_min_resps_received = 0.0
    ctl.gatherer = None
    ctl.gatherer_waiter = MagicMock()
    ctl.trainers = ["site-1"]
    ctl.aggrs = ["site-1"]
    ctl.is_trainer = True
    ctl.is_aggr = True
    ctl.last_aggr_round_done = -1
    ctl.learn_task_abort_timeout = 10.0
    ctl.learn_task_ack_timeout = 10
    ctl.memory_gc_rounds = 1
    ctl.cuda_empty_cache = False
    ctl._aggr_round_count = 0
    ctl.learn_executor = MagicMock()
    ctl.shareable_generator = MagicMock()
    ctl.aggregator = MagicMock()
    ctl.update_status = MagicMock()
    ctl.fire_event = MagicMock()
    ctl.get_config_prop = MagicMock(return_value=1)
    ctl.record_last_result = MagicMock()
    ctl._scatter = MagicMock()
    ctl._distribute_final_results = MagicMock()
    return ctl


class TestHasLazyRefs(unittest.TestCase):
    """_has_lazy_refs() must detect LazyDownloadRef in nested data structures."""

    def test_detects_lazy_ref_in_flat_dict(self):
        data = {"a": LazyDownloadRef("f", "r", "i")}
        self.assertTrue(SwarmClientController._has_lazy_refs(data))

    def test_detects_lazy_ref_in_nested_dict(self):
        data = {"outer": {"inner": LazyDownloadRef("f", "r", "i")}}
        self.assertTrue(SwarmClientController._has_lazy_refs(data))

    def test_detects_lazy_ref_in_shareable(self):
        s = _make_shareable_with_lazy_refs()
        self.assertTrue(SwarmClientController._has_lazy_refs(s))

    def test_no_lazy_ref_in_real_data(self):
        s = _make_shareable_with_real_arrays()
        self.assertFalse(SwarmClientController._has_lazy_refs(s))

    def test_no_lazy_ref_in_empty_dict(self):
        self.assertFalse(SwarmClientController._has_lazy_refs({}))

    def test_no_lazy_ref_in_scalar(self):
        self.assertFalse(SwarmClientController._has_lazy_refs(42))


class TestPrepareLearnTaskData(unittest.TestCase):
    """Swarm owns keep-versus-resolve policy at the learner boundary."""

    @staticmethod
    def _prepare(task_data, learn_executor):
        ctl = _make_controller()
        ctl.learn_executor = learn_executor

        resolve_calls = []
        resolved_result = _make_shareable_with_real_arrays()

        def fake_resolve(res, ctx, **kwargs):
            resolve_calls.append((res, kwargs))
            return resolved_result

        ctl._resolve_lazy_refs = fake_resolve
        fl_ctx = MagicMock()

        controller_data, learner_data = ctl._prepare_learn_task_data(task_data, fl_ctx)
        return resolve_calls, resolved_result, controller_data, learner_data

    def test_external_process_keeps_refs_for_learner(self):
        task_data = _make_shareable_with_lazy_refs()
        executor = ClientAPIExecutor(execution_mode=ExecutionMode.EXTERNAL_PROCESS, command="python train.py")

        resolve_calls, resolved, controller_data, learner_data = self._prepare(task_data, executor)
        self.assertEqual(len(resolve_calls), 1)
        self.assertIs(resolve_calls[0][0], task_data)
        self.assertFalse(resolve_calls[0][1]["enable_tensor_disk_offload"])
        self.assertIs(controller_data, resolved)
        self.assertIs(learner_data, task_data)

    def test_in_process_resolves_refs_for_learner(self):
        task_data = _make_shareable_with_lazy_refs()
        executor = ClientAPIExecutor(execution_mode=ExecutionMode.IN_PROCESS, task_script_path="train.py")

        _, resolved, controller_data, learner_data = self._prepare(task_data, executor)
        self.assertIs(controller_data, resolved)
        self.assertIs(learner_data, resolved)

    def test_non_client_api_executor_is_treated_as_in_process(self):
        task_data = _make_shareable_with_lazy_refs()

        _, resolved, controller_data, learner_data = self._prepare(task_data, MagicMock())
        self.assertIs(controller_data, resolved)
        self.assertIs(learner_data, resolved)

    def test_attach_mode_is_conservatively_treated_as_in_process(self):
        task_data = _make_shareable_with_lazy_refs()
        executor = ClientAPIExecutor(execution_mode=ExecutionMode.ATTACH)

        _, resolved, controller_data, learner_data = self._prepare(task_data, executor)
        self.assertIs(controller_data, resolved)
        self.assertIs(learner_data, resolved)

    def test_real_tensors_need_no_resolution_for_any_executor(self):
        task_data = _make_shareable_with_real_arrays()
        executor = ClientAPIExecutor(execution_mode=ExecutionMode.IN_PROCESS, task_script_path="train.py")

        resolve_calls, _, controller_data, learner_data = self._prepare(task_data, executor)
        self.assertEqual(resolve_calls, [])
        self.assertIs(controller_data, task_data)
        self.assertIs(learner_data, task_data)


class TestResultUploadReceiverStamp(unittest.TestCase):
    def test_swarm_stamps_aggregation_client_fqcn_for_subprocess_result_upload(self):
        ctl = _make_controller()
        task_data = _make_shareable_with_real_arrays()
        fl_ctx = MagicMock()
        fl_ctx.get_job_id.return_value = "job-1"
        aggr_client = MagicMock()
        aggr_client.get_fqcn.return_value = "site-2"
        fl_ctx.get_engine.return_value.get_client_from_name.return_value = aggr_client

        ctl._stamp_result_upload_receiver_ids(task_data, "site-2", fl_ctx)

        self.assertEqual(task_data.get_header(FOBSContextKey.RECEIVER_IDS), ["site-2.job-1"])

    def test_swarm_skips_receiver_stamp_when_job_id_is_unavailable(self):
        ctl = _make_controller()
        task_data = _make_shareable_with_real_arrays()
        fl_ctx = MagicMock()
        fl_ctx.get_job_id.return_value = None

        ctl._stamp_result_upload_receiver_ids(task_data, "site-2", fl_ctx)

        self.assertIsNone(task_data.get_header(FOBSContextKey.RECEIVER_IDS))
        self.assertTrue(ctl.log_warning.called)


class TestScatterLazyRefHandling(unittest.TestCase):
    """_scatter() preserves local refs and opts remote learn tasks into PASS_THROUGH."""

    def _make_real_scatter_ctl(self, me="site-1", trainers=None, aggrs=None):
        ctl = _make_controller()
        del ctl._scatter

        ctl.me = me
        ctl.trainers = trainers or [me]
        ctl.aggrs = aggrs or [me]
        ctl.is_trainer = True
        ctl.is_aggr = True
        ctl.learn_task_timeout = None

        ctl.set_learn_task = MagicMock(return_value=True)
        ctl.send_learn_task = MagicMock(return_value=True)

        from nvflare.app_common.ccwf.common import Constant as _Const

        def cfg(key, *default):
            mapping = {
                _Const.TRAIN_CLIENTS: ctl.trainers,
                _Const.AGGR_CLIENTS: ctl.aggrs,
            }
            return mapping.get(key, default[0] if default else None)

        ctl.get_config_prop = MagicMock(side_effect=cfg)
        return ctl

    def test_local_lazy_task_is_queued_for_mode_aware_preparation(self):
        ctl = self._make_real_scatter_ctl(me="site-1", trainers=["site-1", "site-2"])

        resolve_calls = []
        ctl._resolve_lazy_refs = lambda res, ctx, **kwargs: resolve_calls.append((res, kwargs)) or res
        fl_ctx = MagicMock()

        lazy_task = _make_shareable_with_lazy_refs()
        ctl._scatter(lazy_task, for_round=0, fl_ctx=fl_ctx)

        self.assertEqual(resolve_calls, [])
        local_data = (
            ctl.set_learn_task.call_args.kwargs.get("task_data") or ctl.set_learn_task.call_args[1]["task_data"]
        )
        self.assertTrue(ctl._has_lazy_refs(local_data))

    def test_resolve_skipped_when_real_tensors(self):
        """Sender-is-receiver: aggregator queues locally with real tensors, no resolution."""
        ctl = self._make_real_scatter_ctl(me="site-1", trainers=["site-1", "site-2"])

        resolve_calls = []
        ctl._resolve_lazy_refs = lambda res, ctx, **kwargs: resolve_calls.append((res, kwargs)) or res
        fl_ctx = MagicMock()

        real_task = _make_shareable_with_real_arrays()
        ctl._scatter(real_task, for_round=0, fl_ctx=fl_ctx)

        self.assertEqual(resolve_calls, [], "No resolution needed when task_data has real tensors")

    def test_local_only_lazy_task_is_not_resolved_in_scatter(self):
        ctl = self._make_real_scatter_ctl(me="site-1", trainers=["site-1"])

        resolve_calls = []
        real_data = _make_shareable_with_real_arrays()
        ctl._resolve_lazy_refs = lambda res, ctx, **kwargs: resolve_calls.append((res, kwargs)) or real_data
        fl_ctx = MagicMock()

        lazy_task = _make_shareable_with_lazy_refs()
        ctl._scatter(lazy_task, for_round=0, fl_ctx=fl_ctx)

        self.assertEqual(resolve_calls, [])

    def test_offload_marks_only_remote_learn_task_for_pass_through(self):
        ctl = self._make_real_scatter_ctl(me="site-1", trainers=["site-1", "site-2"])
        ctl.enable_tensor_disk_offload = True
        fl_ctx = MagicMock()

        task_data = _make_shareable_with_real_arrays()
        ctl._scatter(task_data, for_round=0, fl_ctx=fl_ctx)

        remote_data = ctl.send_learn_task.call_args.kwargs["request"]
        local_data = ctl.set_learn_task.call_args.kwargs["task_data"]
        self.assertTrue(remote_data.get_header(ReservedHeaderKey.PASS_THROUGH))
        self.assertIsNone(local_data.get_header(ReservedHeaderKey.PASS_THROUGH))


if __name__ == "__main__":
    unittest.main()

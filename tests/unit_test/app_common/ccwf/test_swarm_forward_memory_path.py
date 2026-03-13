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
"""Tests for receiver-side PASS_THROUGH: _has_lazy_refs detection on SwarmClientController.

Covers:
  1. _has_lazy_refs() correctly detects LazyDownloadRef in nested structures.
  2. _scatter() resolves local copy when LazyDownloadRef is present.
  3. _scatter() skips resolution when task_data has real tensors (sender-is-receiver case).
  4. do_learn_task() GLOBAL_MODEL block: resolves when lazy refs, skips when real tensors.
"""
import unittest
from unittest.mock import MagicMock

import numpy as np

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.shareable import Shareable
from nvflare.app_common.ccwf.swarm_client_ctl import SwarmClientController
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


class TestDoLearnTaskGlobalModel(unittest.TestCase):
    """GLOBAL_MODEL resolution is driven by _has_lazy_refs(), not a flag."""

    def _run_global_model_block(self, task_data):
        ctl = _make_controller()

        resolve_calls = []
        resolved_result = _make_shareable_with_real_arrays()

        def fake_resolve(res, ctx):
            resolve_calls.append(res)
            return resolved_result

        ctl._resolve_lazy_refs = fake_resolve
        model_inputs = []
        ctl.shareable_generator.shareable_to_learnable.side_effect = (
            lambda s, ctx: model_inputs.append(s) or MagicMock()
        )
        fl_ctx = MagicMock()

        task_data_for_model = (
            ctl._resolve_lazy_refs(task_data, fl_ctx) if ctl._has_lazy_refs(task_data) else task_data
        )
        ctl.shareable_generator.shareable_to_learnable(task_data_for_model, fl_ctx)

        return resolve_calls, model_inputs[0] if model_inputs else None

    def test_resolves_when_lazy_refs_present(self):
        task_data = _make_shareable_with_lazy_refs()
        resolve_calls, model_input = self._run_global_model_block(task_data)

        self.assertEqual(len(resolve_calls), 1)
        self.assertIs(resolve_calls[0], task_data)
        self.assertIsNot(model_input, task_data)

    def test_skips_resolution_when_real_tensors(self):
        """Sender-is-receiver case: local queue has real tensors, no resolution needed."""
        task_data = _make_shareable_with_real_arrays()
        resolve_calls, model_input = self._run_global_model_block(task_data)

        self.assertEqual(resolve_calls, [])
        self.assertIs(model_input, task_data)


class TestScatterLazyRefResolution(unittest.TestCase):
    """_scatter() resolves LazyDownloadRefs on local copy based on data content."""

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

    def test_resolve_called_when_lazy_refs_in_task_data(self):
        """When task_data has LazyDownloadRef, _resolve_lazy_refs runs on local copy."""
        ctl = self._make_real_scatter_ctl(me="site-1", trainers=["site-1", "site-2"])

        resolve_calls = []
        real_data = _make_shareable_with_real_arrays()
        ctl._resolve_lazy_refs = lambda res, ctx: resolve_calls.append(res) or real_data
        fl_ctx = MagicMock()

        lazy_task = _make_shareable_with_lazy_refs()
        ctl._scatter(lazy_task, for_round=0, fl_ctx=fl_ctx)

        self.assertEqual(len(resolve_calls), 1)
        local_data = (
            ctl.set_learn_task.call_args.kwargs.get("task_data") or ctl.set_learn_task.call_args[1]["task_data"]
        )
        self.assertIs(local_data, real_data)

    def test_resolve_skipped_when_real_tensors(self):
        """Sender-is-receiver: aggregator queues locally with real tensors, no resolution."""
        ctl = self._make_real_scatter_ctl(me="site-1", trainers=["site-1", "site-2"])

        resolve_calls = []
        ctl._resolve_lazy_refs = lambda res, ctx: resolve_calls.append(res) or res
        fl_ctx = MagicMock()

        real_task = _make_shareable_with_real_arrays()
        ctl._scatter(real_task, for_round=0, fl_ctx=fl_ctx)

        self.assertEqual(resolve_calls, [], "No resolution needed when task_data has real tensors")

    def test_resolve_called_even_when_local_only(self):
        """With lazy refs but no remote targets, resolution still fires (safety)."""
        ctl = self._make_real_scatter_ctl(me="site-1", trainers=["site-1"])

        resolve_calls = []
        real_data = _make_shareable_with_real_arrays()
        ctl._resolve_lazy_refs = lambda res, ctx: resolve_calls.append(res) or real_data
        fl_ctx = MagicMock()

        lazy_task = _make_shareable_with_lazy_refs()
        ctl._scatter(lazy_task, for_round=0, fl_ctx=fl_ctx)

        self.assertEqual(len(resolve_calls), 1, "Lazy refs must be resolved even without remote targets")


if __name__ == "__main__":
    unittest.main()

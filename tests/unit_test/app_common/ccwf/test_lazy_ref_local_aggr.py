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
"""Tests for Fix 17: resolve LazyDownloadRefs before local swarm aggregation.

Covers three scenarios:
  1. _resolve_lazy_refs(): FOBS round-trip with PASS_THROUGH=False in decode context.
  2. Self-aggregation local path: LazyDownloadRefs resolved before _process_learn_result().
  3. Remote P2P path: _resolve_lazy_refs() NOT called (resolution handled by Fix 14).
  4. Defensive guard in _end_gather(): fires, logs error, resolves surviving refs.
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.app_common.ccwf.swarm_client_ctl import SwarmClientController
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef


def _make_shareable_with_lazy_refs():
    """Return a WEIGHT_DIFF Shareable whose values are LazyDownloadRef placeholders."""
    lazy_data = {
        "layer.weight": LazyDownloadRef(fqcn="site-1.subprocess", ref_id="ref-abc", item_id="T0"),
        "layer.bias": LazyDownloadRef(fqcn="site-1.subprocess", ref_id="ref-abc", item_id="T1"),
    }
    dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=lazy_data)
    return dxo.to_shareable()


def _make_shareable_with_real_arrays():
    """Return a WEIGHT_DIFF Shareable with real numpy arrays."""
    real_data = {
        "layer.weight": np.zeros((4, 4), dtype=np.float32),
        "layer.bias": np.zeros((4,), dtype=np.float32),
    }
    dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=real_data)
    return dxo.to_shareable()


def _make_controller():
    """Build a minimal SwarmClientController with all instance attributes set.

    Uses __new__ to skip __init__ (which calls super().__init__ and does component
    lookup), then manually sets every attribute that do_learn_task() and _end_gather()
    read so individual tests don't hit AttributeError.
    """
    ctl = SwarmClientController.__new__(SwarmClientController)
    # logging stubs
    ctl.logger = MagicMock()
    ctl.log_info = MagicMock()
    ctl.log_error = MagicMock()
    ctl.log_debug = MagicMock()
    ctl.log_warning = MagicMock()
    # identity
    ctl.me = "site-1"
    # attributes set by __init__
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
    ctl.forward_pass_through = False
    # component stubs
    ctl.shareable_generator = MagicMock()
    ctl.aggregator = MagicMock()
    ctl.update_status = MagicMock()
    ctl.fire_event = MagicMock()
    ctl.get_config_prop = MagicMock(return_value=1)
    ctl.record_last_result = MagicMock()
    ctl._scatter = MagicMock()
    ctl._distribute_final_results = MagicMock()
    return ctl


class TestResolveRef(unittest.TestCase):
    """Unit tests for _resolve_lazy_refs()."""

    def test_resolve_lazy_refs_returns_original_when_engine_absent(self):
        """If fl_ctx.get_engine() returns None, result is returned unchanged (no crash)."""
        ctl = _make_controller()
        lazy_result = _make_shareable_with_lazy_refs()
        fl_ctx = MagicMock()
        fl_ctx.get_engine.return_value = None
        out = ctl._resolve_lazy_refs(lazy_result, fl_ctx)
        self.assertIs(out, lazy_result)

    def test_resolve_lazy_refs_returns_original_when_cell_absent(self):
        """If engine.get_cell() returns None, result is returned unchanged (no crash)."""
        ctl = _make_controller()
        lazy_result = _make_shareable_with_lazy_refs()
        fl_ctx = MagicMock()
        fl_ctx.get_engine.return_value.get_cell.return_value = None
        out = ctl._resolve_lazy_refs(lazy_result, fl_ctx)
        self.assertIs(out, lazy_result)

    def test_resolve_lazy_refs_returns_original_when_get_cell_missing(self):
        """If engine has no get_cell attribute (e.g. a test stub engine), result is
        returned unchanged — no AttributeError should propagate."""
        ctl = _make_controller()
        lazy_result = _make_shareable_with_lazy_refs()

        # Engine without get_cell (simulates minimal stub engines)
        class _StubEngine:
            pass

        fl_ctx = MagicMock()
        fl_ctx.get_engine.return_value = _StubEngine()
        out = ctl._resolve_lazy_refs(lazy_result, fl_ctx)
        self.assertIs(out, lazy_result)

    def test_resolve_lazy_refs_calls_fobs_round_trip(self):
        """_resolve_lazy_refs() must call fobs.dumps then fobs.loads with PASS_THROUGH=False
        in the decode context supplied by cell.get_fobs_context()."""
        ctl = _make_controller()
        lazy_result = _make_shareable_with_lazy_refs()
        real_result = _make_shareable_with_real_arrays()

        mock_cell = MagicMock()
        fake_decode_ctx = {FOBSContextKey.PASS_THROUGH: False, FOBSContextKey.CELL: mock_cell}
        mock_cell.get_fobs_context.return_value = fake_decode_ctx

        mock_fl_ctx = MagicMock()
        mock_fl_ctx.get_engine.return_value.get_cell.return_value = mock_cell

        with (
            patch("nvflare.fuel.utils.fobs.dumps", return_value=b"encoded") as mock_dumps,
            patch("nvflare.fuel.utils.fobs.loads", return_value=real_result) as mock_loads,
        ):
            out = ctl._resolve_lazy_refs(lazy_result, mock_fl_ctx)

        mock_dumps.assert_called_once_with(lazy_result)
        mock_loads.assert_called_once()

        # cell.get_fobs_context must be called with props containing PASS_THROUGH=False
        mock_cell.get_fobs_context.assert_called_once()
        props = mock_cell.get_fobs_context.call_args.kwargs.get("props", {})
        self.assertFalse(
            props.get(FOBSContextKey.PASS_THROUGH, True), "get_fobs_context must be called with PASS_THROUGH=False"
        )

        # fobs.loads must receive the decode context as fobs_ctx kwarg
        load_kwargs = mock_loads.call_args.kwargs
        self.assertIs(
            load_kwargs.get("fobs_ctx"),
            fake_decode_ctx,
            "fobs.loads must receive the decode context from cell.get_fobs_context()",
        )
        self.assertIs(out, real_result)


class TestLocalAggregationPath(unittest.TestCase):
    """Verify _resolve_lazy_refs() is called before _process_learn_result() for aggr == self.me.

    Rather than invoking the full do_learn_task() (which has many parent-class
    dependencies), we isolate just the local-submit code block by mocking its
    surrounding collaborators and calling do_learn_task() with them all stubbed out.
    """

    def _build_local_aggr_ctl(self):
        """Return a controller wired for local aggregation (aggr == self.me == 'site-1')."""
        ctl = _make_controller()
        ctl.me = "site-1"
        ctl.is_trainer = True

        # Tracks calls to _resolve_lazy_refs and _process_learn_result
        ctl._resolve_calls = []
        ctl._process_calls = []

        real_result = _make_shareable_with_real_arrays()

        def fake_resolve(res, ctx):
            ctl._resolve_calls.append(res)
            return real_result

        def fake_process(req, ctx, sig):
            ctl._process_calls.append(req)
            return make_reply(ReturnCode.OK)

        ctl._resolve_lazy_refs = fake_resolve
        ctl._process_learn_result = fake_process
        return ctl, real_result

    def _make_fl_ctx(self, aggr_site):
        """Return a minimal fl_ctx mock for the given aggregation site."""
        fl_ctx = MagicMock()
        fl_ctx.get_prop.return_value = MagicMock()  # GLOBAL_MODEL is present
        engine = fl_ctx.get_engine.return_value
        engine.new_context.return_value = MagicMock()
        fl_ctx.clone.return_value = MagicMock()
        # Permission request response: granted by the aggr site
        granted = Shareable()
        granted.set_return_code(ReturnCode.OK)
        engine.send_aux_request.return_value = {aggr_site: granted}
        return fl_ctx

    def _make_task_data(self, aggr_site, current_round=0):
        """Return a minimal task_data mock for the given aggregation site."""
        from nvflare.app_common.app_constant import AppConstants
        from nvflare.app_common.ccwf.common import Constant

        headers = {
            Constant.AGGREGATOR: aggr_site,
            AppConstants.CURRENT_ROUND: current_round,
        }
        task_data = MagicMock()
        task_data.get_header.side_effect = lambda k, *a: headers.get(k, a[0] if a else None)
        task_data.set_header = MagicMock()
        task_data.get_cookie_jar.return_value = {}
        return task_data

    def test_local_aggr_resolve_called_before_process(self):
        """When aggr == self.me, _resolve_lazy_refs() is called and the resolved result
        (with real arrays) is what reaches _process_learn_result()."""
        ctl, real_result = self._build_local_aggr_ctl()
        lazy_result = _make_shareable_with_lazy_refs()
        ctl.execute_learn_task = MagicMock(return_value=lazy_result)

        fl_ctx = self._make_fl_ctx(aggr_site="site-1")
        task_data = self._make_task_data(aggr_site="site-1")
        abort_signal = MagicMock()
        abort_signal.triggered = False

        with patch("nvflare.app_common.ccwf.swarm_client_ctl.Gatherer") as MockGatherer:
            MockGatherer.return_value = MagicMock()
            ctl.do_learn_task("learn", task_data, fl_ctx, abort_signal)

        self.assertEqual(len(ctl._resolve_calls), 1, "_resolve_lazy_refs should be called exactly once")
        self.assertIs(
            ctl._resolve_calls[0],
            lazy_result,
            "the lazy result from execute_learn_task must be passed to _resolve_lazy_refs",
        )
        self.assertEqual(len(ctl._process_calls), 1, "_process_learn_result should be called exactly once")
        self.assertIsNot(
            ctl._process_calls[0], lazy_result, "_process_learn_result must receive the resolved (non-lazy) result"
        )
        self.assertIs(
            ctl._process_calls[0],
            real_result,
            "_process_learn_result must receive the real-array result from _resolve_lazy_refs",
        )

    def test_local_aggr_no_resolve_if_execute_fails(self):
        """If execute_learn_task() returns an error RC, _resolve_lazy_refs() must NOT be called."""
        ctl, _ = self._build_local_aggr_ctl()

        err_result = make_reply(ReturnCode.EXECUTION_EXCEPTION)
        ctl.execute_learn_task = MagicMock(return_value=err_result)

        fl_ctx = self._make_fl_ctx(aggr_site="site-1")
        task_data = self._make_task_data(aggr_site="site-1")
        abort_signal = MagicMock()
        abort_signal.triggered = False

        with patch("nvflare.app_common.ccwf.swarm_client_ctl.Gatherer") as MockGatherer:
            MockGatherer.return_value = MagicMock()
            ctl.do_learn_task("learn", task_data, fl_ctx, abort_signal)

        self.assertEqual(ctl._resolve_calls, [], "_resolve_lazy_refs must NOT be called when execute_learn_task fails")


class TestRemotePathUnchanged(unittest.TestCase):
    """For aggr != self.me, _resolve_lazy_refs() must NOT be called on the trainer CJ.
    Resolution for the remote path happens inside broadcast_and_wait() (Fix 14).
    """

    def test_remote_aggr_does_not_call_resolve(self):
        """Remote aggregation path must not eagerly materialise tensors on the trainer CJ."""
        ctl = _make_controller()
        ctl.me = "site-1"
        ctl.is_trainer = True
        ctl.trainers = ["site-1", "site-2"]

        resolve_called = []
        ctl._resolve_lazy_refs = lambda r, c: resolve_called.append(r) or r

        lazy_result = _make_shareable_with_lazy_refs()
        ctl.execute_learn_task = MagicMock(return_value=lazy_result)

        # Wire permission request from site-2 aggr
        fl_ctx = MagicMock()
        fl_ctx.get_prop.return_value = MagicMock()
        engine = fl_ctx.get_engine.return_value
        granted = Shareable()
        granted.set_return_code(ReturnCode.OK)
        engine.send_aux_request.return_value = {"site-2": granted}

        abort_signal = MagicMock()
        abort_signal.triggered = False

        # Remote submit goes through broadcast_and_wait (Fix 14 handles resolution there)
        ok_reply = Shareable()
        ok_reply.set_return_code(ReturnCode.OK)
        ctl.broadcast_and_wait = MagicMock(return_value={"site-2": ok_reply})

        from nvflare.app_common.app_constant import AppConstants
        from nvflare.app_common.ccwf.common import Constant

        headers = {Constant.AGGREGATOR: "site-2", AppConstants.CURRENT_ROUND: 0}
        task_data = MagicMock()
        task_data.get_header.side_effect = lambda k, *a: headers.get(k, a[0] if a else None)
        task_data.set_header = MagicMock()
        task_data.get_cookie_jar.return_value = {}

        # site-1 is not the aggregator so no Gatherer is set up
        with patch("nvflare.app_common.ccwf.swarm_client_ctl.Gatherer"):
            ctl.do_learn_task("learn", task_data, fl_ctx, abort_signal)

        self.assertEqual(resolve_called, [], "_resolve_lazy_refs must NOT be called for remote aggregator path")
        ctl.broadcast_and_wait.assert_called_once()


class TestDefensiveGuardInEndGather(unittest.TestCase):
    """_end_gather() defensive check must log error and resolve surviving LazyDownloadRefs."""

    def _build_end_gather_ctl(self, num_rounds=10):
        """Return a controller ready for _end_gather() testing.

        num_rounds > for_round ensures the else-branch runs (next round starts),
        which calls learnable_to_shareable and _scatter.  Set num_rounds=1 to
        exercise the 'training done' branch instead.
        """
        ctl = _make_controller()

        def cfg(key, *default):
            mapping = {
                "start_round": 0,
                "num_rounds": num_rounds,
            }
            return mapping.get(key, default[0] if default else None)

        ctl.get_config_prop = MagicMock(side_effect=cfg)
        # learnable_to_shareable must return a real Shareable to pass the assert
        ctl.shareable_generator.learnable_to_shareable.return_value = _make_shareable_with_real_arrays()
        return ctl

    def test_defensive_guard_fires_and_resolves_lazy_refs(self):
        """If LazyDownloadRefs survive into _end_gather(), the guard resolves them and
        logs an error, and shareable_to_learnable receives the resolved result."""
        ctl = self._build_end_gather_ctl()

        lazy_aggr = _make_shareable_with_lazy_refs()
        real_aggr = _make_shareable_with_real_arrays()

        resolve_calls = []

        def fake_resolve(res, ctx):
            resolve_calls.append(res)
            return real_aggr

        ctl._resolve_lazy_refs = fake_resolve

        mock_gatherer = MagicMock()
        mock_gatherer.aggregate.return_value = lazy_aggr
        mock_gatherer.for_round = 0
        mock_gatherer.fl_ctx = MagicMock()

        ctl._end_gather(mock_gatherer)

        self.assertEqual(len(resolve_calls), 1, "defensive guard must call _resolve_lazy_refs exactly once")
        ctl.log_error.assert_called()  # the unexpected path must be logged as an error
        called_with = ctl.shareable_generator.shareable_to_learnable.call_args[0][0]
        self.assertIs(called_with, real_aggr, "shareable_to_learnable must receive the resolved result, not lazy refs")

    def test_defensive_guard_does_not_interfere_with_real_arrays(self):
        """When aggr_result already contains real arrays, the guard must not call
        _resolve_lazy_refs() and must not call log_error for this path."""
        ctl = self._build_end_gather_ctl()

        real_aggr = _make_shareable_with_real_arrays()
        resolve_calls = []

        def fake_resolve(res, ctx):
            resolve_calls.append(res)
            return res

        ctl._resolve_lazy_refs = fake_resolve

        mock_gatherer = MagicMock()
        mock_gatherer.aggregate.return_value = real_aggr
        mock_gatherer.for_round = 0
        mock_gatherer.fl_ctx = MagicMock()

        ctl._end_gather(mock_gatherer)

        # Guard must not activate for clean (non-lazy) aggregated results
        self.assertEqual(resolve_calls, [], "_resolve_lazy_refs must NOT be called when result has real arrays")


if __name__ == "__main__":
    unittest.main()

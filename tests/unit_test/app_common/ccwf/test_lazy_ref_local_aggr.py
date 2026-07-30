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
"""Tests for resolving LazyDownloadRefs at the Swarm aggregation boundary.

Covers four scenarios:
  1. _resolve_lazy_refs(): FOBS round-trip with PASS_THROUGH=False in decode context.
  2. Self-aggregation local path: LazyDownloadRefs resolved before _process_learn_result().
  3. Remote P2P path: trainer CJ preserves refs for explicit aggregation-CJ resolution.
  4. Defensive guard in _end_gather(): fires, calls system_panic (invariant violation).
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.swarm_client_ctl import SwarmClientController
from nvflare.app_common.utils.tensor_disk_offload_context import _TENSOR_DISK_OFFLOAD_ROOT_DIR
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef


def _make_shareable_with_lazy_refs(relay=False):
    """Return a WEIGHT_DIFF Shareable whose values are LazyDownloadRef placeholders."""
    lazy_data = {
        "layer.weight": LazyDownloadRef(fqcn="site-1.subprocess", ref_id="ref-abc", item_id="T0", relay=relay),
        "layer.bias": LazyDownloadRef(fqcn="site-1.subprocess", ref_id="ref-abc", item_id="T1", relay=relay),
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
    ctl.enable_tensor_disk_offload = True
    ctl._tensor_disk_offload_root_dir = "/tmp/swarm-offload"
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
    # component stubs
    ctl.shareable_generator = MagicMock()
    ctl.aggregator = MagicMock()
    ctl.update_status = MagicMock()
    ctl.fire_event = MagicMock()
    ctl.get_config_prop = MagicMock(return_value=1)
    ctl.record_last_result = MagicMock()
    ctl._scatter = MagicMock()
    ctl._distribute_final_results = MagicMock()
    ctl.system_panic = MagicMock()
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
        """Relay refs must be encoded with a Cell and decoded with PASS_THROUGH=False."""
        ctl = _make_controller()
        lazy_result = _make_shareable_with_lazy_refs(relay=True)
        real_result = _make_shareable_with_real_arrays()

        mock_cell = MagicMock()
        fake_encode_ctx = {FOBSContextKey.CELL: mock_cell}
        fake_decode_ctx = {FOBSContextKey.PASS_THROUGH: False, FOBSContextKey.CELL: mock_cell}
        mock_cell.get_fobs_context.side_effect = [fake_encode_ctx, fake_decode_ctx]

        mock_fl_ctx = MagicMock()
        mock_fl_ctx.get_engine.return_value.get_cell.return_value = mock_cell

        with (
            patch("nvflare.fuel.utils.fobs.dumps", return_value=b"encoded") as mock_dumps,
            patch("nvflare.fuel.utils.fobs.loads", return_value=real_result) as mock_loads,
        ):
            out = ctl._resolve_lazy_refs(lazy_result, mock_fl_ctx)

        mock_dumps.assert_called_once_with(lazy_result, fobs_ctx=fake_encode_ctx)
        mock_loads.assert_called_once()

        # Encode and decode need separate contexts because FOBS mutates its context
        # with operation-local state.
        self.assertEqual(mock_cell.get_fobs_context.call_count, 2)
        self.assertIsNot(fake_encode_ctx, fake_decode_ctx)
        props = mock_cell.get_fobs_context.call_args_list[1].kwargs.get("props", {})
        self.assertFalse(
            props.get(FOBSContextKey.PASS_THROUGH, True), "get_fobs_context must be called with PASS_THROUGH=False"
        )
        self.assertTrue(
            props.get(FOBSContextKey.TENSOR_DISK_OFFLOAD, False),
            "local aggregation result resolution must opt in to tensor disk offload",
        )
        self.assertEqual(
            props.get(_TENSOR_DISK_OFFLOAD_ROOT_DIR),
            "/tmp/swarm-offload",
            "the terminal resolution must carry the workflow-owned root explicitly",
        )

        # fobs.loads must receive the decode context as fobs_ctx kwarg
        load_kwargs = mock_loads.call_args.kwargs
        self.assertIs(
            load_kwargs.get("fobs_ctx"),
            fake_decode_ctx,
            "fobs.loads must receive the decode context from cell.get_fobs_context()",
        )
        self.assertIs(out, real_result)

    def test_resolve_lazy_refs_can_disable_disk_offload_for_trainer_input(self):
        ctl = _make_controller()
        lazy_result = _make_shareable_with_lazy_refs()
        mock_cell = MagicMock()
        mock_cell.get_fobs_context.return_value = {}
        mock_fl_ctx = MagicMock()
        mock_fl_ctx.get_engine.return_value.get_cell.return_value = mock_cell

        with (
            patch("nvflare.fuel.utils.fobs.dumps", return_value=b"encoded"),
            patch("nvflare.fuel.utils.fobs.loads", return_value=_make_shareable_with_real_arrays()),
        ):
            ctl._resolve_lazy_refs(lazy_result, mock_fl_ctx, enable_tensor_disk_offload=False)

        props = mock_cell.get_fobs_context.call_args.kwargs["props"]
        self.assertFalse(props[FOBSContextKey.TENSOR_DISK_OFFLOAD])


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

    def test_external_process_local_aggr_resolves_lazy_result_before_process(self):
        """An external-process trainer/aggregator resolves its local lazy result."""
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

    def test_in_process_local_aggr_keeps_materialized_result_in_memory(self):
        """An in-process local result bypasses the LazyDownloadRef resolution path."""
        ctl, _ = self._build_local_aggr_ctl()
        in_memory_result = _make_shareable_with_real_arrays()
        ctl.execute_learn_task = MagicMock(return_value=in_memory_result)

        fl_ctx = self._make_fl_ctx(aggr_site="site-1")
        task_data = self._make_task_data(aggr_site="site-1")
        abort_signal = MagicMock()
        abort_signal.triggered = False

        with patch("nvflare.app_common.ccwf.swarm_client_ctl.Gatherer") as mock_gatherer:
            mock_gatherer.return_value = MagicMock()
            ctl.do_learn_task("learn", task_data, fl_ctx, abort_signal)

        self.assertEqual(ctl._resolve_calls, [])
        self.assertEqual(ctl._process_calls, [in_memory_result])

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


class TestRemoteAggregationPath(unittest.TestCase):
    """For aggr != self.me, _resolve_lazy_refs() must NOT be called on the trainer CJ.
    The result is marked PASS_THROUGH so resolution happens explicitly on the
    aggregation client's CJ.
    """

    def test_aggregation_client_resolves_passed_through_result_before_gather(self):
        class _FakeGatherer:
            for_round = 0

            def __init__(self):
                self.gathered_result = None

            def gather(self, _client_name, request, _fl_ctx):
                self.gathered_result = request
                return make_reply(ReturnCode.OK)

        ctl = _make_controller()
        ctl.me = "site-2"
        ctl.gatherer = _FakeGatherer()

        lazy_result = _make_shareable_with_lazy_refs()
        lazy_result.set_header(AppConstants.CURRENT_ROUND, 0)
        lazy_result.set_header(ReservedHeaderKey.PASS_THROUGH, True)
        resolved_result = _make_shareable_with_real_arrays()
        resolve_calls = []

        def resolve(result, fl_ctx):
            resolve_calls.append(result)
            return resolved_result

        ctl._resolve_lazy_refs = resolve
        fl_ctx = MagicMock()
        peer_ctx = FLContext()
        peer_ctx.set_prop(ReservedKey.IDENTITY_NAME, "site-1", private=True, sticky=False)
        fl_ctx.get_peer_context.return_value = peer_ctx
        abort_signal = MagicMock()
        abort_signal.triggered = False

        with patch("nvflare.app_common.ccwf.swarm_client_ctl.Gatherer", _FakeGatherer):
            reply = ctl._process_learn_result(lazy_result, fl_ctx, abort_signal)

        self.assertEqual(reply.get_return_code(), ReturnCode.OK)
        self.assertEqual(resolve_calls, [lazy_result])
        gathered_result = ctl.gatherer.gathered_result
        self.assertIs(gathered_result, resolved_result)
        self.assertFalse(gathered_result.get_header(ReservedHeaderKey.PASS_THROUGH))

    def test_aggregation_client_clears_pass_through_without_lazy_refs(self):
        class _FakeGatherer:
            for_round = 0

            def __init__(self):
                self.gathered_result = None

            def gather(self, _client_name, request, _fl_ctx):
                self.gathered_result = request
                return make_reply(ReturnCode.OK)

        ctl = _make_controller()
        ctl.me = "site-2"
        ctl.gatherer = _FakeGatherer()
        ctl._resolve_lazy_refs = MagicMock()

        result = _make_shareable_with_real_arrays()
        result.set_header(AppConstants.CURRENT_ROUND, 0)
        result.set_header(ReservedHeaderKey.PASS_THROUGH, True)
        fl_ctx = MagicMock()
        peer_ctx = FLContext()
        peer_ctx.set_prop(ReservedKey.IDENTITY_NAME, "site-1", private=True, sticky=False)
        fl_ctx.get_peer_context.return_value = peer_ctx
        abort_signal = MagicMock()
        abort_signal.triggered = False

        with patch("nvflare.app_common.ccwf.swarm_client_ctl.Gatherer", _FakeGatherer):
            reply = ctl._process_learn_result(result, fl_ctx, abort_signal)

        self.assertEqual(reply.get_return_code(), ReturnCode.OK)
        ctl._resolve_lazy_refs.assert_not_called()
        self.assertFalse(ctl.gatherer.gathered_result.get_header(ReservedHeaderKey.PASS_THROUGH))

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

        # The sending CJ preserves the refs; the remote aggregation controller
        # resolves them after receiving the submission.
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
        sent_result = ctl.broadcast_and_wait.call_args.kwargs["task"].data
        self.assertTrue(sent_result.get_header(ReservedHeaderKey.PASS_THROUGH))

    def test_in_process_result_is_forwarded_to_remote_aggregator_without_local_offload(self):
        ctl = _make_controller()
        ctl.me = "site-1"
        ctl.is_trainer = True
        ctl.trainers = ["site-1", "site-2"]

        resolve_called = []
        ctl._resolve_lazy_refs = lambda result, ctx: resolve_called.append(result) or result
        in_memory_result = _make_shareable_with_real_arrays()
        ctl.execute_learn_task = MagicMock(return_value=in_memory_result)

        fl_ctx = MagicMock()
        fl_ctx.get_prop.return_value = MagicMock()
        granted = make_reply(ReturnCode.OK)
        fl_ctx.get_engine.return_value.send_aux_request.return_value = {"site-2": granted}
        ctl.broadcast_and_wait = MagicMock(return_value={"site-2": make_reply(ReturnCode.OK)})

        abort_signal = MagicMock()
        abort_signal.triggered = False
        task_data = self._make_remote_task_data()

        ctl.do_learn_task("learn", task_data, fl_ctx, abort_signal)

        self.assertEqual(resolve_called, [])
        sent_result = ctl.broadcast_and_wait.call_args.kwargs["task"].data
        self.assertIs(sent_result, in_memory_result)
        self.assertTrue(sent_result.get_header(ReservedHeaderKey.PASS_THROUGH))

    @staticmethod
    def _make_remote_task_data():
        from nvflare.app_common.app_constant import AppConstants
        from nvflare.app_common.ccwf.common import Constant

        headers = {Constant.AGGREGATOR: "site-2", AppConstants.CURRENT_ROUND: 0}
        task_data = MagicMock()
        task_data.get_header.side_effect = lambda key, *default: headers.get(key, default[0] if default else None)
        task_data.set_header = MagicMock()
        task_data.get_cookie_jar.return_value = {}
        return task_data


class TestDefensiveGuardInEndGather(unittest.TestCase):
    """_end_gather() defensive check must call system_panic when LazyDownloadRefs survive.

    YT1 fix: reaching _end_gather() with unresolved LazyDownloadRef objects is a code
    invariant violation — _resolve_lazy_refs() should have been called upstream on the
    local-aggregation path (_scatter() or do_learn_task()).  Logging and recovering would
    mask the root cause; system_panic + return is the correct response.
    """

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

    def test_defensive_guard_fires_system_panic_on_lazy_refs(self):
        """If LazyDownloadRefs survive into _end_gather(), call system_panic
        and return early.  Reaching this point is a code bug — _resolve_lazy_refs()
        was not called on the local-aggregation path where it should have been."""
        ctl = self._build_end_gather_ctl()

        lazy_aggr = _make_shareable_with_lazy_refs()

        resolve_calls = []
        ctl._resolve_lazy_refs = lambda res, ctx: resolve_calls.append(res) or res

        mock_gatherer = MagicMock()
        mock_gatherer.aggregate.return_value = lazy_aggr
        mock_gatherer.for_round = 0
        mock_gatherer.fl_ctx = MagicMock()

        ctl._end_gather(mock_gatherer)

        # Must panic — not recover
        ctl.system_panic.assert_called_once()
        msg = ctl.system_panic.call_args[0][0]
        assert "LazyDownloadRef" in msg, f"panic message must name the type, got: {msg}"
        # Must return early — no resolution, no further processing
        self.assertEqual(resolve_calls, [], "_resolve_lazy_refs must NOT be called after system_panic")
        ctl.shareable_generator.shareable_to_learnable.assert_not_called()

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

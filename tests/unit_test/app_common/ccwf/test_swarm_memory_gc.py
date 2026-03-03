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

"""Tests for SwarmClientController per-round aggregator GC cadence."""

from unittest.mock import Mock, patch

from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.common import Constant
from nvflare.app_common.ccwf.swarm_client_ctl import SwarmClientController


def _make_controller(memory_gc_rounds=1, cuda_empty_cache=False):
    """Create a minimal SwarmClientController without calling __init__."""
    ctrl = SwarmClientController.__new__(SwarmClientController)
    ctrl.memory_gc_rounds = memory_gc_rounds
    ctrl.cuda_empty_cache = cuda_empty_cache
    ctrl._aggr_round_count = 0
    ctrl.shareable_generator = Mock()
    ctrl.shareable_generator.shareable_to_learnable.return_value = Mock()
    ctrl.shareable_generator.learnable_to_shareable.return_value = Shareable()
    ctrl.forward_pass_through = False
    ctrl.record_last_result = Mock()
    ctrl._distribute_final_results = Mock()
    ctrl._scatter = Mock()
    ctrl.log_error = Mock()
    ctrl.log_info = Mock()
    ctrl.log_debug = Mock()
    ctrl.update_status = Mock()
    return ctrl


def _make_gatherer(for_round=0):
    """Create a mock Gatherer."""
    gatherer = Mock()
    gatherer.aggregate.return_value = Shareable()
    gatherer.for_round = for_round
    gatherer.fl_ctx = Mock()
    return gatherer


def _call_end_gather(ctrl, gatherer, num_rounds_total=5):
    """Call _end_gather with get_config_prop mocked."""

    def get_config_prop(key, default=None):
        if key == Constant.START_ROUND:
            return 0
        if key == AppConstants.NUM_ROUNDS:
            return num_rounds_total
        return default

    ctrl.get_config_prop = get_config_prop
    ctrl._end_gather(gatherer)


class TestSwarmAggregatorMemoryGC:
    """Test per-round GC cadence in SwarmClientController._end_gather."""

    def test_gc_disabled_when_memory_gc_rounds_zero(self):
        """cleanup_memory is never called when memory_gc_rounds=0."""
        ctrl = _make_controller(memory_gc_rounds=0)

        with patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup:
            _call_end_gather(ctrl, _make_gatherer(for_round=0))
            mock_cleanup.assert_not_called()

    def test_gc_fires_every_round_when_memory_gc_rounds_one(self):
        """cleanup_memory is called every round when memory_gc_rounds=1 (legacy behavior)."""
        ctrl = _make_controller(memory_gc_rounds=1)

        with patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup:
            for r in range(3):
                _call_end_gather(ctrl, _make_gatherer(for_round=r))
            assert mock_cleanup.call_count == 3

    def test_gc_fires_every_n_rounds(self):
        """cleanup_memory fires every N rounds when memory_gc_rounds=N."""
        ctrl = _make_controller(memory_gc_rounds=2)

        with patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup:
            for r in range(4):
                _call_end_gather(ctrl, _make_gatherer(for_round=r))
            # rounds 2 and 4 fire; rounds 1 and 3 do not
            assert mock_cleanup.call_count == 2

    def test_gc_passes_cuda_empty_cache_false(self):
        """cuda_empty_cache=False is forwarded to cleanup_memory."""
        ctrl = _make_controller(memory_gc_rounds=1, cuda_empty_cache=False)

        with patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup:
            _call_end_gather(ctrl, _make_gatherer(for_round=0))
            mock_cleanup.assert_called_once_with(cuda_empty_cache=False)

    def test_gc_passes_cuda_empty_cache_true(self):
        """cuda_empty_cache=True is forwarded to cleanup_memory (swarm client has GPU)."""
        ctrl = _make_controller(memory_gc_rounds=1, cuda_empty_cache=True)

        with patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup:
            _call_end_gather(ctrl, _make_gatherer(for_round=0))
            mock_cleanup.assert_called_once_with(cuda_empty_cache=True)

    def test_gc_fires_on_final_round(self):
        """cleanup_memory fires on the final round (not skipped at training end)."""
        ctrl = _make_controller(memory_gc_rounds=1)
        # for_round=4, num_rounds_total=5 → final round → _distribute_final_results path
        with patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup:
            _call_end_gather(ctrl, _make_gatherer(for_round=4), num_rounds_total=5)
            mock_cleanup.assert_called_once()

    def test_gc_not_disabled_skips_intermediate_rounds(self):
        """With memory_gc_rounds=3, only every 3rd round fires GC."""
        ctrl = _make_controller(memory_gc_rounds=3)

        with patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup:
            for r in range(6):
                _call_end_gather(ctrl, _make_gatherer(for_round=r))
            # rounds 3 and 6 fire
            assert mock_cleanup.call_count == 2


class TestSwarmAggregatorGCLogging:
    """M3 fix: _end_gather() must log at INFO when cleanup_memory() fires.

    Before M3 fix, cleanup_memory() was called silently — no log message.
    Unlike all other GC call sites (CJ-side), Swarm aggregator GC was invisible
    in runtime logs, making it impossible to verify GC execution from logs alone.
    """

    def test_gc_fires_log_info_called(self):
        """When memory_gc_rounds=1 and GC fires, log_info must be called."""
        ctrl = _make_controller(memory_gc_rounds=1)

        with patch("nvflare.fuel.utils.memory_utils.cleanup_memory"):
            _call_end_gather(ctrl, _make_gatherer(for_round=0))

        ctrl.log_info.assert_called()
        msgs = [str(call_args) for call_args in ctrl.log_info.call_args_list]
        assert any(
            "cleanup" in m.lower() or "memory" in m.lower() or "aggr" in m.lower() for m in msgs
        ), f"log_info must mention cleanup/memory/aggr, got calls: {msgs}"

    def test_gc_skipped_round_no_cleanup_log(self):
        """When GC does not fire on a given round, no cleanup log_info is emitted."""
        ctrl = _make_controller(memory_gc_rounds=2)

        with patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup:
            _call_end_gather(ctrl, _make_gatherer(for_round=0))  # round 1 — GC skipped
            mock_cleanup.assert_not_called()

        # Only non-GC log_info calls should have been made (round completion etc.)
        # None of them should mention "cleanup" since GC did not fire.
        for call_args in ctrl.log_info.call_args_list:
            msg = str(call_args)
            assert "cleanup" not in msg.lower(), f"cleanup log must not appear when GC is skipped, got: {msg}"

    def test_gc_log_mentions_round_count(self):
        """Log message must include the aggregation round count."""
        ctrl = _make_controller(memory_gc_rounds=1)

        with patch("nvflare.fuel.utils.memory_utils.cleanup_memory"):
            _call_end_gather(ctrl, _make_gatherer(for_round=0))

        all_msgs = " ".join(str(c) for c in ctrl.log_info.call_args_list)
        assert "1" in all_msgs, "GC log message must include the round count (1 after first GC call)"

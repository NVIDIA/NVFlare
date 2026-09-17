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

"""Test-only Swarm components that make disk offload observable."""

import json
import os
import threading

from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.swarm_client_ctl import SwarmClientController
from nvflare.app_opt.pt.lazy_tensor_dict import _LazyRef

OFFLOAD_OBSERVATIONS_FILE = "tensor_offload_observations.jsonl"


def _get_run_dir(fl_ctx: FLContext) -> str:
    engine = fl_ctx.get_engine()
    if not engine:
        raise RuntimeError("engine is missing from FLContext")
    return engine.get_workspace().get_run_dir(fl_ctx.get_job_id())


class RotatingSwarmClientController(SwarmClientController):
    """Select site-1 for round 0 and site-2 for round 1."""

    def _scatter(self, task_data: Shareable, for_round: int, fl_ctx: FLContext) -> bool:
        candidates = self.aggrs
        if not candidates:
            return super()._scatter(task_data, for_round, fl_ctx)

        selected = candidates[for_round % len(candidates)]
        self.aggrs = [selected]
        self.log_info(fl_ctx, f"TENSOR_OFFLOAD_AGGREGATOR round={for_round} site={selected}")
        try:
            return super()._scatter(task_data, for_round, fl_ctx)
        finally:
            self.aggrs = candidates


class TensorOffloadValidatingAggregator(InTimeAccumulateWeightedAggregator):
    """Record on-disk lazy refs before the production aggregator materializes them."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._observation_lock = threading.Lock()

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        dxo = from_shareable(shareable)
        data = dxo.data
        values = list(data.values()) if hasattr(data, "values") else []
        lazy_refs = [value for value in values if isinstance(value, _LazyRef)]
        if not values or len(lazy_refs) != len(values):
            raise RuntimeError(
                f"expected every aggregation tensor to be disk-backed; "
                f"got {len(lazy_refs)} lazy refs for {len(values)} values"
            )

        file_paths = [ref.file_path for ref in lazy_refs]
        missing_paths = [path for path in file_paths if not os.path.isfile(path)]
        if missing_paths:
            raise RuntimeError(f"tensor offload files are missing before aggregation: {missing_paths}")

        root_dirs = sorted({os.path.dirname(os.path.dirname(path)) for path in file_paths})
        if any(not os.path.basename(root).startswith("nvflare_tensor_offload_") for root in root_dirs):
            raise RuntimeError(f"unexpected tensor offload roots: {root_dirs}")

        observation = {
            "site": fl_ctx.get_identity_name(),
            "round": fl_ctx.get_prop(AppConstants.CURRENT_ROUND),
            "contributor": shareable.get_peer_prop(ReservedKey.IDENTITY_NAME, "?"),
            "file_paths": file_paths,
            "root_dirs": root_dirs,
            "total_bytes": sum(os.path.getsize(path) for path in file_paths),
        }
        marker_path = os.path.join(_get_run_dir(fl_ctx), OFFLOAD_OBSERVATIONS_FILE)
        with self._observation_lock:
            with open(marker_path, "a", encoding="utf-8") as marker:
                marker.write(json.dumps(observation, sort_keys=True) + "\n")

        self.log_info(
            fl_ctx,
            f"TENSOR_OFFLOAD_ACTIVE site={observation['site']} round={observation['round']} "
            f"files={len(file_paths)} bytes={observation['total_bytes']}",
        )
        return super().accept(shareable, fl_ctx)

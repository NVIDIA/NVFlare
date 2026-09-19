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
"""NVFlare persistor for strict adapter checkpoints and round artifacts."""

import json
import os

import adapter_checkpoint

from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor


class AdapterPTFileModelPersistor(PTFileModelPersistor):
    def __init__(self, *args, adapter_identity: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_identity = dict(adapter_identity or {})

    def save_model(self, ml: ModelLearnable, fl_ctx):
        manager = self._get_persistence_manager(fl_ctx)
        manager.update(ml)
        state = adapter_checkpoint.strip_model_prefix(manager.var_dict)
        state = adapter_checkpoint.align_adapter_state_strict(state, state)
        manifest = adapter_checkpoint.build_adapter_manifest(state, identity=self.adapter_identity)
        manager.other_props[adapter_checkpoint.ADAPTER_MANIFEST_KEY] = manifest
        if manager.meta is None:
            manager.meta = {}
        manager.meta[adapter_checkpoint.ADAPTER_MANIFEST_KEY] = manifest
        self.save_model_file(self._ckpt_save_path)

        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        if current_round is None:
            return
        round_dir = os.path.join(self.log_dir, "server_rounds", f"round_{int(current_round)}")
        os.makedirs(round_dir, exist_ok=True)
        round_checkpoint = os.path.join(round_dir, self.global_model_file_name)
        self.save_model_file(round_checkpoint)
        round_manifest = {
            "schema_version": 1,
            "round": int(current_round),
            "aggregate_adapter_hash": manifest["adapter_hash"],
            "tensor_count": manifest["tensor_count"],
            "checkpoint_location": os.path.abspath(round_checkpoint),
            "aggregation_stats": adapter_checkpoint.metadata_safe(
                fl_ctx.get_prop(AppConstants.AGGREGATION_STATS) or {}
            ),
        }
        with open(os.path.join(round_dir, "round_manifest.json"), "w") as f:
            json.dump(round_manifest, f, indent=2, sort_keys=True)

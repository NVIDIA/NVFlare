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

"""lerobot adapter — bridges HuggingFace lerobot policies to decent-vla.

Covers any policy class registered with `lerobot.policies.factory.make_policy`
(pi0, pi05, smolvla, ACT, VQBeT, …). Requires the `[lerobot]` install extra:

    pip install -e ".[lerobot]"

The adapter accepts a path to an existing lerobot `TrainPipelineConfig`
YAML in `backend.extra.train_config_path`; this lets you reuse configs
you already have rather than re-expressing every lerobot field in
decent-vla's schema. See `examples/smolvla_libero_fl_fedavg.yaml`.
"""

from __future__ import annotations

from decent_vla.adapters.lerobot.backend import LeRobotBackend
from decent_vla.adapters.lerobot.policy import LeRobotPolicy

__all__ = ["LeRobotBackend", "LeRobotPolicy"]

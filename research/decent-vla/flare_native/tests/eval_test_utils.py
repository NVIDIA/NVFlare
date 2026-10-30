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


from pathlib import Path


def _write_train_eval_cfg(
    tmp_path: Path,
    fname: str,
    num_rounds: int,
    output_dir: Path,
    local_steps: int = 2,
    suites: list[str] | None = None,
    benchmark_extra: dict | None = None,
) -> Path:
    import yaml

    cfg = {
        "name": "prov_run",
        "device": "cpu",
        "backend": {"name": "mock"},
        "data": {"name": "mock", "batch_size": 4, "num_clients": 2, "partitioning": "iid"},
        "fl": {
            "strategy": {"name": "fedavg"},
            "num_rounds": num_rounds,
            "local_steps": local_steps,
            "save_every_n_rounds": 1,
            "mode": "local",
            "output_dir": str(output_dir),
        },
        "eval": {
            "benchmark": {"name": "mock", "suites": suites or ["default"], "extra": benchmark_extra or {}},
            "n_episodes_per_task": 2,
            "seed": 0,
        },
    }
    p = tmp_path / fname
    p.write_text(yaml.safe_dump(cfg))
    return p

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

"""Generated recipes retain the policy-specific training and preprocessing settings."""

import importlib.util
from pathlib import Path

import pytest
import yaml

from src.configs.loader import load_run_config

spec = importlib.util.spec_from_file_location("vla_job", Path(__file__).resolve().parents[2] / "job.py")
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)
staged_job = job.staged_job


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize(
    "policy,tuning,algorithm,rounds,batch,workers,lr",
    [
        ("pi05", "fullft", "fedavg", 100, 32, 2, 2.5e-5),
        ("pi05", "fullft", "fedprox", 100, 32, 2, 2.5e-5),
        ("pi05", "fullft", "fedadam", 100, 32, 2, 2.5e-5),
        ("pi05", "lora", "fedavg", 100, 32, 4, 2.5e-4),
        ("smolvla", "fullft", "fedavg", 160, 32, 12, 1e-4),
        ("smolvla", "lora", "fedavg", 160, 64, 8, 1e-4),
        ("groot", "fullft", "fedavg", 140, 16, 16, 1e-4),
        ("groot", "lora", "fedavg", 140, 32, 16, 1e-4),
    ],
)
def test_paper_recipes(tmp_path, policy, tuning, algorithm, rounds, batch, workers, lr, smoke):
    args = job.define_parser().parse_args(
        ["--policy", policy, "--tuning", tuning, "--algorithm", algorithm, "--output-dir", str(tmp_path)]
        + (["--smoke"] if smoke else [])
    )
    path = job.write_config(args)
    cfg = load_run_config(path)
    assert cfg.fl.num_rounds == (2 if smoke else rounds)
    assert cfg.fl.local_steps == (20 if smoke else 250)
    assert cfg.data.num_clients == (2 if smoke else 8)
    assert (cfg.data.batch_size, cfg.data.num_workers, cfg.backend.optim.lr) == (batch, workers, lr)
    if policy != "groot":
        pipeline = yaml.safe_load(Path(cfg.backend.extra["train_config_path"]).read_text())["policy"]
        assert pipeline["type"] == policy
        if policy == "smolvla" and tuning == "fullft":
            assert pipeline["train_expert_only"] is False and pipeline["freeze_vision_encoder"] is False
            assert cfg.backend.optim.grad_clip_norm == 10
    if algorithm == "fedprox":
        assert cfg.fl.strategy.mu == 0.01 and cfg.fl.strategy.extra["normalize"] is False
    if algorithm == "fedadam":
        assert (cfg.fl.strategy.server_lr, cfg.fl.strategy.server_eps) == (0.001, 0.001)


def test_generated_pipeline_is_packaged_for_deployment(tmp_path):
    args = job.define_parser().parse_args(["--policy", "pi05", "--smoke", "--output-dir", str(tmp_path)])
    path = job.write_config(args)
    with staged_job(str(path)) as (stage, files):
        raw = yaml.safe_load((stage / "run.yaml").read_text())
        name = raw["backend"]["extra"]["train_config_path"]
        assert name == "train_pipeline.yaml" and name in files
        assert yaml.safe_load((stage / name).read_text())["policy"]["empty_cameras"] == 1

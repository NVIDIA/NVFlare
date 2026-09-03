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

import importlib.util
import shlex
from pathlib import Path
from types import SimpleNamespace


def _load_qwen_job_module():
    path = Path(__file__).resolve().parents[3] / "examples" / "advanced" / "qwen3-vl" / "job.py"
    spec = importlib.util.spec_from_file_location("qwen3_vl_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qwen_job_uses_nvflare_torchrun_wrapper():
    module = _load_qwen_job_module()

    assert shlex.split(module._build_torchrun_command(4)) == [
        "python3",
        "-m",
        "nvflare.app_opt.pt.torchrun_node",
        "--nproc-per-node=4",
    ]


def test_qwen_train_args_preserve_paths_with_spaces(tmp_path):
    module = _load_qwen_job_module()
    args = SimpleNamespace(
        model_name_or_path="/models/Qwen VL",
        max_steps=10,
        learning_rate="5e-7",
        lora=True,
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.1,
    )

    train_args = module._build_train_args(
        args,
        str(tmp_path / "site data" / "site-1"),
        str(tmp_path / "image root"),
        "none",
    )

    assert shlex.split(train_args) == [
        "--data_path",
        str(tmp_path / "site data" / "site-1"),
        "--image_root",
        str(tmp_path / "image root"),
        "--dataset_use",
        "fl_site",
        "--model_name_or_path",
        "/models/Qwen VL",
        "--max_steps",
        "10",
        "--learning_rate",
        "5e-7",
        "--report_to",
        "none",
        "--lora_exchange",
        "--lora_r",
        "64",
        "--lora_alpha",
        "128",
        "--lora_dropout",
        "0.1",
    ]

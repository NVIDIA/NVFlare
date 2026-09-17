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
        lora=False,
    )
    site_data_path = str(tmp_path / "site data" / "site-1")
    image_root = str(tmp_path / "image root")

    parts = shlex.split(module._build_train_args(args, site_data_path, image_root, "none"))

    assert parts[parts.index("--data_path") + 1] == site_data_path
    assert parts[parts.index("--image_root") + 1] == image_root
    assert parts[parts.index("--model_name_or_path") + 1] == "/models/Qwen VL"
    assert parts[parts.index("--max_steps") + 1] == "10"

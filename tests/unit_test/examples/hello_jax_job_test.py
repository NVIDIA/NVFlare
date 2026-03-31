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
import os
import shlex
from types import SimpleNamespace


def _load_job_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    module_path = os.path.join(repo_root, "examples", "hello-world", "hello-jax", "job.py")
    spec = importlib.util.spec_from_file_location("hello_jax_job", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_train_args_quotes_data_dir_with_spaces():
    job_module = _load_job_module()
    args = SimpleNamespace(
        epochs=1,
        batch_size=128,
        learning_rate=0.05,
        momentum=0.9,
        n_clients=2,
        data_dir="/tmp/nvflare data/hello jax/mnist",
    )

    train_args = job_module._build_train_args(args)

    assert shlex.split(train_args) == [
        "--epochs",
        "1",
        "--batch_size",
        "128",
        "--learning_rate",
        "0.05",
        "--momentum",
        "0.9",
        "--num_partitions",
        "2",
        "--data_dir",
        "/tmp/nvflare data/hello jax/mnist",
    ]

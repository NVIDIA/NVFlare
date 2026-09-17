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
from pathlib import Path


def _load_prepare_data_module():
    repo_root = Path(__file__).parents[3]
    module_path = repo_root / "examples" / "hello-world" / "hello-lightning" / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("hello_lightning_prepare_data", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_data_downloads_train_and_test_splits(monkeypatch, tmp_path):
    module = _load_prepare_data_module()
    calls = []

    def fake_cifar10(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(module.datasets, "CIFAR10", fake_cifar10)

    module.prepare_data(tmp_path)

    assert calls == [
        {"root": tmp_path, "train": True, "download": True},
        {"root": tmp_path, "train": False, "download": True},
    ]

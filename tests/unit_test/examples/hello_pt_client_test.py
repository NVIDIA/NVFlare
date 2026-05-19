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
import sys

import pytest

HAS_PT_DEPS = all(importlib.util.find_spec(dep) is not None for dep in ("torch", "torchvision"))
pytestmark = pytest.mark.skipif(not HAS_PT_DEPS, reason="PyTorch example dependencies are not installed")


def _load_hello_pt_module(file_name: str, module_name: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    example_dir = os.path.join(repo_root, "examples", "hello-world", "hello-pt")
    module_path = os.path.join(example_dir, file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    original_model_module = sys.modules.pop("model", None)
    sys.path.insert(0, example_dir)
    try:
        spec.loader.exec_module(module)
    except RuntimeError as e:
        if "torchvision" in str(e):
            pytest.skip(f"PyTorch example dependency is unavailable: {e}")
        raise
    finally:
        sys.path.pop(0)
        if original_model_module is not None:
            sys.modules["model"] = original_model_module
        else:
            sys.modules.pop("model", None)
    return module


def test_hello_pt_evaluate_rejects_empty_data_loader():
    client_module = _load_hello_pt_module("client.py", "hello_pt_client")

    with pytest.raises(ValueError, match="Evaluation data_loader produced no samples"):
        client_module.evaluate(net=None, data_loader=[], device="cpu")

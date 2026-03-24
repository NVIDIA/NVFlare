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

import numpy as np
import pytest

HAS_JAX_DEPS = all(importlib.util.find_spec(dep) is not None for dep in ("jax", "flax", "optax"))
pytestmark = pytest.mark.skipif(not HAS_JAX_DEPS, reason="JAX example dependencies are not installed")


def _load_model_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    model_path = os.path.join(repo_root, "examples", "hello-world", "hello-jax", "model.py")
    spec = importlib.util.spec_from_file_location("hello_jax_model", model_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_jax_param_flatten_roundtrip():
    model_module = _load_model_module()
    params = model_module.create_initial_params()

    flat_params = model_module.flatten_params(params)
    restored_params = model_module.unflatten_params(flat_params)
    restored_flat_params = model_module.flatten_params(restored_params)

    assert isinstance(flat_params, np.ndarray)
    assert flat_params.ndim == 1
    np.testing.assert_allclose(flat_params, restored_flat_params, rtol=1e-6, atol=1e-6)


def test_jax_train_state_uses_same_param_structure():
    model_module = _load_model_module()
    params = model_module.create_initial_params()
    state = model_module.create_train_state(params, learning_rate=0.05, momentum=0.9)

    flat_params = model_module.flatten_params(params)
    flat_state_params = model_module.flatten_params(state.params)
    np.testing.assert_allclose(flat_params, flat_state_params, rtol=1e-6, atol=1e-6)

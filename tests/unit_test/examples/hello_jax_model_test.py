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

import numpy as np
import pytest

HAS_JAX_DEPS = all(importlib.util.find_spec(dep) is not None for dep in ("jax", "flax", "optax"))
pytestmark = pytest.mark.skipif(not HAS_JAX_DEPS, reason="JAX example dependencies are not installed")


def _load_hello_jax_module(file_name: str, module_name: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    example_dir = os.path.join(repo_root, "examples", "hello-world", "hello-jax")
    module_path = os.path.join(example_dir, file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, example_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_jax_param_flatten_roundtrip():
    model_module = _load_hello_jax_module("model.py", "hello_jax_model")
    params = model_module.create_initial_params()

    flat_params = model_module.flatten_params(params)
    restored_params = model_module.unflatten_params(flat_params)
    restored_flat_params = model_module.flatten_params(restored_params)

    assert isinstance(flat_params, np.ndarray)
    assert flat_params.ndim == 1
    np.testing.assert_allclose(flat_params, restored_flat_params, rtol=1e-6, atol=1e-6)


def test_jax_train_state_uses_same_param_structure():
    model_module = _load_hello_jax_module("model.py", "hello_jax_model")
    params = model_module.create_initial_params()
    state = model_module.create_train_state(params, learning_rate=0.05, momentum=0.9)

    flat_params = model_module.flatten_params(params)
    flat_state_params = model_module.flatten_params(state.params)
    np.testing.assert_allclose(flat_params, flat_state_params, rtol=1e-6, atol=1e-6)


def test_jax_train_epoch_rejects_empty_data():
    client_module = _load_hello_jax_module("client.py", "hello_jax_client")
    model_module = _load_hello_jax_module("model.py", "hello_jax_model")
    params = model_module.create_initial_params()
    state = model_module.create_train_state(params, learning_rate=0.05, momentum=0.9)
    empty_images = np.zeros((0, 28, 28, 1), dtype=np.float32)
    empty_labels = np.zeros((0,), dtype=np.int32)

    with pytest.raises(ValueError, match="No training data available"):
        client_module.train_epoch(state, empty_images, empty_labels, 128, client_module.jax.random.PRNGKey(0))


def test_jax_evaluate_rejects_empty_data():
    client_module = _load_hello_jax_module("client.py", "hello_jax_client")
    model_module = _load_hello_jax_module("model.py", "hello_jax_model")
    params = model_module.create_initial_params()
    empty_images = np.zeros((0, 28, 28, 1), dtype=np.float32)
    empty_labels = np.zeros((0,), dtype=np.int32)

    with pytest.raises(ValueError, match="No evaluation data available"):
        client_module.evaluate(params, empty_images, empty_labels, 128)


def test_split_for_client_rejects_out_of_range_partition():
    client_module = _load_hello_jax_module("client.py", "hello_jax_client")
    images = np.arange(4)
    labels = np.arange(4)

    with pytest.raises(ValueError, match="exceeds available partitions"):
        client_module.split_for_client(images, labels, "site-3", 2)

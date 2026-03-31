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
import types

import numpy as np
import pytest


def _load_client_module_without_jax():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    module_path = os.path.join(repo_root, "examples", "hello-world", "hello-jax", "client.py")

    fake_jax = types.ModuleType("jax")
    fake_jnp = types.ModuleType("jax.numpy")
    fake_jax.jit = lambda fn: fn
    fake_jax.numpy = fake_jnp

    fake_optax = types.ModuleType("optax")
    fake_optax.softmax_cross_entropy_with_integer_labels = lambda *args, **kwargs: None

    fake_model = types.ModuleType("model")
    fake_model.MODEL = types.SimpleNamespace(apply=lambda *args, **kwargs: None)
    fake_model.create_train_state = lambda *args, **kwargs: None
    fake_model.flatten_params = lambda *args, **kwargs: None
    fake_model.unflatten_params = lambda *args, **kwargs: None

    overrides = {
        "jax": fake_jax,
        "jax.numpy": fake_jnp,
        "optax": fake_optax,
        "model": fake_model,
    }
    original_modules = {name: sys.modules.get(name) for name in overrides}

    try:
        sys.modules.update(overrides)
        spec = importlib.util.spec_from_file_location("hello_jax_client_utils", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_split_for_client_rejects_out_of_range_partition():
    client_module = _load_client_module_without_jax()
    images = np.arange(4)
    labels = np.arange(4)

    with pytest.raises(ValueError, match="exceeds available partitions"):
        client_module.split_for_client(images, labels, "site-3", 2)


def test_split_for_client_rejects_non_positive_site_index():
    client_module = _load_client_module_without_jax()
    images = np.arange(4)
    labels = np.arange(4)

    with pytest.raises(ValueError, match="1-indexed site numbering"):
        client_module.split_for_client(images, labels, "site-0", 2)

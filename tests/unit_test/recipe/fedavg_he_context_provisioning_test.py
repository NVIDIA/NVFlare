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

import pytest
import torch.nn as nn

from nvflare.recipe.sim_env import SimEnv

pytestmark = pytest.mark.skipif(importlib.util.find_spec("tenseal") is None, reason="tenseal is not installed")


class SimpleTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 10)

    def forward(self, x):
        return self.lin(x)


def _create_recipe(train_script: str):
    from nvflare.app_opt.pt.recipes.fedavg_he import FedAvgRecipeWithHE

    return FedAvgRecipeWithHE(
        name="fedavg-he-test",
        model=SimpleTestModel(),
        train_script=train_script,
        min_clients=2,
        num_rounds=1,
    )


def test_process_env_raises_descriptive_error_when_contexts_missing(tmp_path):
    from nvflare.app_opt.pt.recipes.fedavg_he import HE_CONTEXT_PROVISIONING_DOC_LINK

    train_script = tmp_path / "client.py"
    train_script.write_text("print('train')\n")
    recipe = _create_recipe(str(train_script))
    env = SimEnv(num_clients=2, workspace_root=str(tmp_path))

    with pytest.raises(ValueError) as exc_info:
        recipe.process_env(env)

    err = str(exc_info.value)
    assert "TenSEAL contexts must be generated before running HE jobs." in err
    assert HE_CONTEXT_PROVISIONING_DOC_LINK in err
    assert "server_context.tenseal" in err
    assert "client_context.tenseal" in err


def test_process_env_passes_when_contexts_exist(tmp_path):
    train_script = tmp_path / "client.py"
    train_script.write_text("print('train')\n")
    recipe = _create_recipe(str(train_script))
    env = SimEnv(num_clients=2, workspace_root=str(tmp_path))

    startup_dir = tmp_path / recipe.name / "startup"
    startup_dir.mkdir(parents=True, exist_ok=True)
    (startup_dir / "server_context.tenseal").write_bytes(b"server")
    (startup_dir / "client_context.tenseal").write_bytes(b"client")

    recipe.process_env(env)

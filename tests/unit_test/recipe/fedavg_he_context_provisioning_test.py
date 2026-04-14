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
from unittest.mock import MagicMock

import pytest

try:
    import torch.nn as nn
except ImportError:
    nn = None

from nvflare.recipe.sim_env import SimEnv

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tenseal") is None or nn is None,
    reason="tenseal and torch are required",
)

if nn is not None:

    class SimpleTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(10, 10)

        def forward(self, x):
            return self.lin(x)

else:

    class SimpleTestModel:
        pass


def _create_recipe(train_script: str):
    from nvflare.app_opt.pt.recipes.fedavg_he import FedAvgRecipeWithHE

    return FedAvgRecipeWithHE(
        name="fedavg-he-test",
        model=SimpleTestModel(),
        train_script=train_script,
        min_clients=2,
        num_rounds=1,
    )


def test_process_env_rejects_sim_env_for_he_recipe(tmp_path):
    from nvflare.app_opt.pt.recipes.fedavg_he import HE_CONTEXT_PROVISIONING_DOC_LINK

    train_script = tmp_path / "client.py"
    train_script.write_text("print('train')\n")
    recipe = _create_recipe(str(train_script))
    env = SimEnv(num_clients=2, workspace_root=str(tmp_path))

    with pytest.raises(ValueError) as exc_info:
        recipe.process_env(env)

    err = str(exc_info.value)
    assert "FedAvgRecipeWithHE does not support SimEnv." in err
    assert "HEBuilder" in err
    assert HE_CONTEXT_PROVISIONING_DOC_LINK in err


def test_process_env_allows_non_sim_env(tmp_path):
    from nvflare.recipe.spec import ExecEnv

    train_script = tmp_path / "client.py"
    train_script.write_text("print('train')\n")
    recipe = _create_recipe(str(train_script))

    non_sim_env = MagicMock(spec=ExecEnv)
    # Should not raise any exception for a non-SimEnv environment
    recipe.process_env(non_sim_env)

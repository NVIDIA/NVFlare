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


"""Native LearnablePersistor checkpoint/resume contract."""

from types import SimpleNamespace

import torch
import yaml

from src.persistor import VLAPersistor


def test_native_checkpoint_restores_weights_round_and_adam_moments(tmp_path):
    raw = {
        "name": "resume",
        "device": "cpu",
        "backend": {"name": "mock"},
        "data": {"num_clients": 2},
        "fl": {"num_rounds": 4, "local_steps": 2, "output_dir": str(tmp_path / "out"), "strategy": {"name": "fedadam"}},
    }
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(raw))
    workspace = SimpleNamespace(get_app_config_dir=lambda jid: str(tmp_path), get_run_dir=lambda jid: str(tmp_path))
    ctx = SimpleNamespace(
        get_engine=lambda: SimpleNamespace(get_workspace=lambda: workspace),
        get_job_id=lambda: "test",
        get_prop=lambda key: 0,
    )
    persistor = VLAPersistor("run.yaml")
    model = persistor.load(ctx)
    param = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.Adam([param], lr=0.1)
    param.grad = torch.ones(2)
    opt.step()
    persistor.optimizer_state = opt.state_dict()
    persistor.save(model, ctx)
    resumed = VLAPersistor("run.yaml")
    restored = resumed.load(ctx)
    assert resumed.start_round == 1
    assert next(iter(resumed.optimizer_state["state"].values()))["step"] == 1
    for key in model["weights"]:
        torch.testing.assert_close(torch.as_tensor(restored["weights"][key]), torch.as_tensor(model["weights"][key]))

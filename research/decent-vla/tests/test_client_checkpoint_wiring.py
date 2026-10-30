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

"""Wiring tests: config flag parsing + LocalRunner client-checkpoint save path
(opt-in fl.persist_client_state). The build_client_state RESTORE path itself is
covered by test_client_checkpoint.py.
"""
from __future__ import annotations

import torch
import yaml

from decent_vla.configs import load_run_config
from decent_vla.runtime.client_checkpoint import latest_client_checkpoint
from decent_vla.runtime.local_runner import LocalRunner
from decent_vla.strategies.fedavg import FedAvg


def test_config_parses_persist_client_state(tmp_path):
    # default is OFF (no behavior change)
    assert load_run_config("examples/mock_fl_fedavg.yaml").fl.persist_client_state is False
    # and the flag is honored when set (not silently ignored)
    raw = yaml.safe_load(open("examples/mock_fl_fedavg.yaml"))
    raw["fl"]["persist_client_state"] = True
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(raw))
    assert load_run_config(str(p)).fl.persist_client_state is True


def test_localrunner_writes_client_checkpoints_when_enabled(
    tmp_path, backend_factory, data_cfg, backend_cfg
):
    ckpt = tmp_path / "client_states"
    runner = LocalRunner(
        backend_factory=backend_factory, strategy=FedAvg(None),
        client_ids=["c0", "c1"], partition_indices=[list(range(16)), list(range(16, 32))],
        data_cfg=data_cfg, num_rounds=2, local_steps=3, optim_cfg=backend_cfg.optim,
        device=torch.device("cpu"), save_every_n_rounds=1,
        persist_client_state=True, client_ckpt_dir=str(ckpt),
    )
    runner.run()
    assert latest_client_checkpoint(ckpt, "c0") is not None
    assert latest_client_checkpoint(ckpt, "c1") is not None


def test_localrunner_no_checkpoints_when_disabled(
    tmp_path, backend_factory, data_cfg, backend_cfg
):
    ckpt = tmp_path / "client_states"
    runner = LocalRunner(
        backend_factory=backend_factory, strategy=FedAvg(None),
        client_ids=["c0"], partition_indices=[list(range(16))],
        data_cfg=data_cfg, num_rounds=1, local_steps=2, optim_cfg=backend_cfg.optim,
        device=torch.device("cpu"),  # persist off (default)
    )
    runner.run()
    assert latest_client_checkpoint(ckpt, "c0") is None

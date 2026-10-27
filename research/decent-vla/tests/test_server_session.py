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

"""write_final_artifacts: export order and the CPU fallback on a full GPU."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from decent_vla.runtime import server_session
from decent_vla.runtime.round_loop import RoundLoopOutcome


def _setup(monkeypatch, tmp_path, fail_on_gpu: bool):
    devices = []

    class _Backend:
        def set_state(self, state, scope):
            self.state = state

        def save_checkpoint(self, path, spec):
            # The resume blob must not exist yet: it is written after the export.
            assert not (tmp_path / "checkpoints" / "server_round_final.pt").exists()
            torch.save(self.state, path)

    def _factory_for(cfg, device):
        def _factory():
            devices.append(device.type)
            if fail_on_gpu and device.type != "cpu":
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return _Backend()

        return _factory, _Backend

    monkeypatch.setattr(server_session, "build_backend_factory", _factory_for)
    monkeypatch.setattr(server_session, "resolve_device", lambda _d: torch.device("cuda"))
    cfg = SimpleNamespace(device="auto", fl=SimpleNamespace(output_dir=str(tmp_path), num_rounds=2))
    outcome = RoundLoopOutcome(final_state={"w": torch.ones(2)}, rounds_completed=2, per_round_metrics=[])
    return cfg, outcome, devices


@pytest.mark.parametrize("fail_on_gpu, expected_devices", [(False, ["cuda"]), (True, ["cuda", "cpu"])])
def test_final_export_then_resume_blob(monkeypatch, tmp_path, fail_on_gpu, expected_devices):
    cfg, outcome, devices = _setup(monkeypatch, tmp_path, fail_on_gpu)
    ckpt = server_session.write_final_artifacts(cfg, outcome)
    assert ckpt == tmp_path / "checkpoints" / "round_0002.pt" and ckpt.exists()
    blob = torch.load(tmp_path / "checkpoints" / "server_round_final.pt", weights_only=True)
    assert blob["round"] == 1 and torch.equal(blob["state"]["w"], torch.ones(2))
    assert devices == expected_devices

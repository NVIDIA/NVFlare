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

"""MockPolicy — the inference wrapper paired with MockBackend."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MockPolicy:
    def __init__(self, model: nn.Module, in_dim: int, device: torch.device):
        self._model = model.eval()
        self._in_dim = in_dim
        self._device = device
        self._rng = np.random.default_rng(0)

    def reset(self, task_meta: dict) -> None:
        self._rng = np.random.default_rng(int(task_meta.get("seed", 0)))

    def select_action(self, observation: dict) -> np.ndarray:
        # Canonical observation (design §6.1): the MLP consumes the state
        # vector. Fall back to a random observation for ad-hoc callers.
        ob = observation.get("state")
        if ob is None:
            ob = self._rng.standard_normal(self._in_dim).astype(np.float32)
        x = torch.from_numpy(np.asarray(ob, dtype=np.float32)).to(self._device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            y = self._model(x).cpu().numpy()
        return y[0]

    @property
    def action_horizon(self) -> int:
        return 1

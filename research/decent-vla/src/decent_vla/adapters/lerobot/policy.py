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

"""LeRobotPolicy — inference wrapper for any `PreTrainedPolicy`.

This is what `LeRobotBackend.build_policy()` returns. It implements the
`decent_vla.core.policy.Policy` protocol by delegating to the lerobot
policy's `select_action(observation)` method.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


class LeRobotPolicy:
    """Thin wrapper around a lerobot PreTrainedPolicy for inference.

    Benchmarks call `policy.reset(task_meta)` at episode start and
    `policy.select_action(observation)` per timestep.
    """

    def __init__(self, policy: Any, device: torch.device, preprocessor: Any = None, postprocessor: Any = None, autocast_dtype: Any = None):
        self._policy = policy
        self._device = device
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._autocast_dtype = autocast_dtype
        self._policy.eval()

    def reset(self, task_meta: dict) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def select_action(self, observation: dict) -> np.ndarray:
        # Canonical observation (design §6.1) from a benchmark: build the
        # lerobot batch here. Anything else is treated as a pre-built batch
        # (legacy/advanced use) and passed through unchanged.
        canonical = "images" in observation and "state" in observation
        if canonical:
            obs = self._batch_from_canonical(observation)
        else:
            obs = observation
        # Preprocess: if the policy was loaded with a preprocessor (normalization,
        # rename, device transfer), apply it. Otherwise pass through.
        if self._preprocessor is not None:
            obs = self._preprocessor(obs)
        elif not canonical:
            obs = {k: _to_tensor(v, self._device) for k, v in obs.items()}
        with torch.no_grad():
            if self._autocast_dtype is not None:
                with torch.autocast(device_type=self._device.type, dtype=self._autocast_dtype):
                    action = self._policy.select_action(obs)
            else:
                action = self._policy.select_action(obs)
        if self._postprocessor is not None:
            action = self._postprocessor(action)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action)
        if canonical and action.ndim > 1 and action.shape[0] == 1:
            # contract: 1-D env-ready action; lerobot returns a [1, D] batch
            action = action[0]
        return action

    def _batch_from_canonical(self, observation: dict) -> dict:
        """Canonical observation -> lerobot batch. Mirrors
        `lerobot.envs.utils.preprocess_observation` + task injection:
        HWC uint8 [0,255] -> [1,3,H,W] float32 [0,1], state -> [1,D]
        float32, instruction -> `task` list."""
        batch: dict[str, Any] = {}
        for name, img in observation["images"].items():
            t = torch.from_numpy(np.ascontiguousarray(img))
            if t.ndim != 3 or not (t.shape[2] < t.shape[0] and t.shape[2] < t.shape[1]):
                raise ValueError(
                    f"image {name!r} must be HWC channel-last, got shape {tuple(t.shape)}"
                )
            if t.dtype != torch.uint8:
                raise ValueError(f"image {name!r} must be uint8, got {t.dtype}")
            t = t.permute(2, 0, 1).unsqueeze(0).to(torch.float32).div_(255)
            batch[f"observation.images.{name}"] = t.to(self._device)
        state = torch.as_tensor(observation["state"], dtype=torch.float32)
        batch["observation.state"] = state.unsqueeze(0).to(self._device)
        batch["task"] = [observation["instruction"]]
        return batch

    @property
    def action_horizon(self) -> int:
        # Most lerobot policies expose either `config.n_action_steps` or
        # `config.action_horizon`; fall back to 1.
        cfg = getattr(self._policy, "config", None)
        for attr in ("n_action_steps", "action_horizon"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and v > 0:
                return v
        return 1


def _to_tensor(v: Any, device: torch.device) -> Any:
    if isinstance(v, torch.Tensor):
        return v.to(device)
    if isinstance(v, np.ndarray):
        return torch.from_numpy(v).to(device)
    return v

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

"""LeRobotPolicy canonical-observation handling (design §6.1) — CPU only,
no lerobot import: the wrapper itself only needs torch/numpy."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from decent_vla.adapters.lerobot.policy import LeRobotPolicy


class _FakeInner:
    """Stands in for a lerobot PreTrainedPolicy."""

    def __init__(self, action):
        self._action = action
        self.seen = None

    def eval(self):
        return self

    def reset(self):
        pass

    def select_action(self, batch):
        self.seen = batch
        return self._action


def _canonical():
    img = np.zeros((8, 8, 3), np.uint8)
    img[0, 0] = (255, 128, 0)
    return {
        "images": {"image": img, "image2": np.full((8, 8, 3), 51, np.uint8)},
        "state": np.arange(8, dtype=np.float32),
        "instruction": "pick up the bowl",
    }


def test_canonical_obs_builds_lerobot_batch():
    """Mirrors lerobot's preprocess_observation: HWC uint8 -> [1,3,H,W]
    float32 in [0,1], state -> [1,D] float32, instruction -> task list."""
    captured = {}

    def preprocessor(batch):
        captured.update(batch)
        return batch

    inner = _FakeInner(torch.zeros(1, 7))
    policy = LeRobotPolicy(inner, torch.device("cpu"), preprocessor=preprocessor)
    policy.select_action(_canonical())

    img = captured["observation.images.image"]
    assert img.shape == (1, 3, 8, 8) and img.dtype == torch.float32
    assert torch.isclose(img[0, 0, 0, 0], torch.tensor(1.0))  # R of marker, /255
    assert torch.isclose(img[0, 1, 0, 0], torch.tensor(128 / 255))
    img2 = captured["observation.images.image2"]
    assert torch.allclose(img2, torch.full((1, 3, 8, 8), 51 / 255))
    state = captured["observation.state"]
    assert state.shape == (1, 8) and state.dtype == torch.float32
    assert captured["task"] == ["pick up the bowl"]


def test_canonical_obs_action_squeezed_to_1d():
    """The contract action is 1-D env-ready; lerobot returns a [1, D] batch."""
    inner = _FakeInner(torch.arange(7, dtype=torch.float32).reshape(1, 7))
    policy = LeRobotPolicy(inner, torch.device("cpu"))
    action = policy.select_action(_canonical())
    assert action.shape == (7,)
    np.testing.assert_allclose(action, np.arange(7, dtype=np.float32))


def test_canonical_obs_rejects_non_uint8_image():
    inner = _FakeInner(torch.zeros(1, 7))
    policy = LeRobotPolicy(inner, torch.device("cpu"))
    obs = _canonical()
    obs["images"]["image"] = obs["images"]["image"].astype(np.float32)
    with pytest.raises(ValueError, match="uint8"):
        policy.select_action(obs)


def test_legacy_prebuilt_batch_passthrough():
    """Non-canonical dicts keep the historical behavior: handed to the
    policy as-is (tensorized when there is no preprocessor)."""
    inner = _FakeInner(torch.zeros(1, 7))
    policy = LeRobotPolicy(inner, torch.device("cpu"))
    policy.select_action({"observation": torch.randn(1, 16)})
    assert set(inner.seen) == {"observation"}

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

"""GrootPolicy — inference wrapper so decent-vla can eval GR00T checkpoints.

``backend.py`` handles GR00T training (load model+processor, FL finetune); this
module adds the inference path the decent-vla eval loop needs.
``GrootBackend.build_policy()`` returns a ``GrootPolicy`` with the
interface used by evaluation::

    reset(task_meta)           -> None              # episode start
    select_action(observation) -> np.ndarray        # 1-D, env-ready action
    action_horizon             -> int

It mirrors the upstream reference ``gr00t/policy/gr00t_policy.py::Gr00tPolicy``
(``_get_action``) but drives the **already-loaded, FL-aggregated** in-memory
model + processor instead of re-loading from a checkpoint dir, and bridges the
decent-vla *canonical observation* (design §6.1: images HWC uint8, 1-D state,
instruction str) into GR00T's nested ``{video, state, language}`` modality
format (keys/splits from the embodiment modality config + dataset modality.json).

GR00T predicts an action **chunk** (horizon = ``len(action.delta_indices)``; 16
for N1.7/LIBERO). The benchmark consumes one action per ``select_action``, so we
buffer the chunk and pop one per call, re-planning every ``replan_steps`` steps
(default = full horizon, GR00T's intended open-loop chunk execution; override
via ``backend.extra.eval_replan_steps``).

GR00T-specific imports stay function-local so this module imports cleanly
without Isaac-GR00T installed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


class GrootPolicy:
    """Inference over an in-memory Gr00tN1d7 model and processor."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        embodiment_value: str,
        modality_configs: dict,
        device: torch.device,
        state_index_ranges: dict[str, tuple[int, int]],
        action_index_ranges: dict[str, tuple[int, int]],
        video_key_map: dict[str, str],
        autocast_dtype: torch.dtype | None = None,
        replan_steps: int | None = None,
        gripper_transform: str = "normalize_invert",
    ) -> None:
        from gr00t.data.embodiment_tags import EmbodimentTag

        self._model = model
        self._model.eval()
        self._processor = processor
        if hasattr(self._processor, "eval"):
            self._processor.eval()  # disables train-time state dropout
        self._device = device
        self._autocast_dtype = autocast_dtype

        self._embodiment_tag = EmbodimentTag.resolve(embodiment_value)
        self._modality_configs = modality_configs
        self._state_ranges = state_index_ranges
        self._action_ranges = action_index_ranges
        self._video_key_map = dict(video_key_map)  # groot video key -> canonical image key

        self._language_key = modality_configs["language"].modality_keys[0]
        self._state_keys = list(modality_configs["state"].modality_keys)
        self._action_keys = list(modality_configs["action"].modality_keys)
        self._action_dim = max(hi for _, hi in action_index_ranges.values())

        self._horizon = len(modality_configs["action"].delta_indices)  # 16 for N1.7/LIBERO
        if not replan_steps or replan_steps <= 0:
            replan_steps = self._horizon
        self._replan_steps = min(int(replan_steps), self._horizon)
        self._gripper_transform = str(gripper_transform)

        self._queue: list[np.ndarray] = []
        log.info(
            "GrootPolicy ready: emb=%s horizon=%d replan=%d action_dim=%d video=%s",
            embodiment_value,
            self._horizon,
            self._replan_steps,
            self._action_dim,
            self._video_key_map,
        )

    # -- evaluation interface ----------------------------------------------

    def reset(self, task_meta: dict) -> None:
        self._queue.clear()

    def select_action(self, observation: dict) -> np.ndarray:
        if not self._queue:
            chunk = self._predict_chunk(observation)  # (horizon, action_dim) float32
            n = min(self._replan_steps, len(chunk))
            self._queue = [np.asarray(chunk[i], dtype=np.float32) for i in range(n)]
        return self._queue.pop(0)

    @property
    def action_horizon(self) -> int:
        return self._replan_steps

    # -- inference -------------------------------------------------------

    def _predict_chunk(self, observation: dict) -> np.ndarray:
        from gr00t.data.types import MessageType, VLAStepData

        video, states = self._canonical_to_groot(observation)  # per-sample, T=1
        vla = VLAStepData(
            images=video,
            states=states,
            actions={},  # no ground-truth actions at inference
            text=str(observation["instruction"]),
            embodiment=self._embodiment_tag,
        )
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla}]
        collated = self._processor.collator([self._processor(messages)])
        collated = _prepare(collated, self._device, torch.bfloat16)

        with torch.inference_mode():
            if self._autocast_dtype is not None:
                with torch.autocast(device_type=self._device.type, dtype=self._autocast_dtype):
                    pred = self._model.get_action(**collated)
            else:
                pred = self._model.get_action(**collated)
        normalized = pred["action_pred"].float().cpu().numpy()  # (1, horizon, action_dim)

        batched_states = {k: np.stack([states[k]], axis=0) for k in self._state_keys}  # (1,T,D)
        unnorm = self._processor.decode_action(normalized, self._embodiment_tag, batched_states)

        # decode_action -> {action_key: (1, horizon, d_key)}; reassemble (horizon, action_dim)
        chunk = np.zeros((self._horizon, self._action_dim), dtype=np.float32)
        for key in self._action_keys:
            lo, hi = self._action_ranges[key]
            arr = np.asarray(unnorm[key], dtype=np.float32)[0]  # (horizon, d_key)
            chunk[:, lo:hi] = arr[: self._horizon, : hi - lo]
        # GR00T outputs gripper in RLDS convention ([0,1], 0=close/1=open); the
        # LIBERO env expects [-1,+1] with -1=open/+1=close. Mirror the reference
        # gr00t/eval/sim/LIBERO/libero_env.py (normalize_gripper_action +
        # invert_gripper_action). Gated by backend.extra.eval_gripper_transform.
        if self._gripper_transform == "normalize_invert" and "gripper" in self._action_ranges:
            glo, ghi = self._action_ranges["gripper"]
            g = np.sign(2.0 * chunk[:, glo:ghi] - 1.0)  # [0,1] -> {-1,+1} (binarize)
            chunk[:, glo:ghi] = -g  # invert for env convention
        return chunk

    def _canonical_to_groot(self, observation: dict) -> tuple[dict, dict]:
        """canonical obs -> (video, state) in GR00T per-sample (T=1) form.

        video[groot_key] = (T=1, H, W, 3) uint8   (collator adds the batch dim)
        state[state_key] = (T=1, d_key)   float32  (split per dataset modality.json)
        """
        imgs = observation["images"]
        video: dict[str, np.ndarray] = {}
        for groot_key, canon_key in self._video_key_map.items():
            if canon_key not in imgs:
                raise KeyError(
                    f"canonical observation has no image {canon_key!r} for GR00T video "
                    f"key {groot_key!r}; available images: {sorted(imgs)}"
                )
            im = np.ascontiguousarray(imgs[canon_key])
            if im.dtype != np.uint8 or im.ndim != 3 or im.shape[-1] != 3:
                raise ValueError(f"image {canon_key!r} must be HWC uint8 RGB, got {im.dtype} shape {im.shape}")
            video[groot_key] = im[None]  # (T=1, H, W, 3)

        state_vec = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
        states: dict[str, np.ndarray] = {}
        for key in self._state_keys:
            lo, hi = self._state_ranges[key]
            states[key] = state_vec[lo:hi].reshape(1, hi - lo).astype(np.float32)  # (T=1, d_key)
        return video, states


def _prepare(x: Any, device: torch.device, dtype: torch.dtype) -> Any:
    """Recursively move tensors to ``device`` and cast floating tensors to ``dtype``.

    Mirrors gr00t_policy._rec_to_dtype but also pins device (the collator yields
    CPU tensors; the model lives on ``device``).
    """
    if isinstance(x, torch.Tensor):
        x = x.to(device)
        return x.to(dtype=dtype) if torch.is_floating_point(x) else x
    if hasattr(x, "items"):
        return {k: _prepare(v, device, dtype) for k, v in x.items()}
    if isinstance(x, list):
        return [_prepare(v, device, dtype) for v in x]
    return x

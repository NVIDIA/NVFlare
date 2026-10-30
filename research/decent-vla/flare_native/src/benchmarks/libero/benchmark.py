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

"""LIBERO benchmark adapter (spec §7). Fresh sim rollouts; evaluates ALL
tasks of each requested suite regardless of the training pool.

Imports of the `libero` package are lazy (inside methods): `libero` comes
from lerobot's [libero] extra or a clone on PYTHONPATH (LIBERO_DIR).
Rollout validation requires the real simulation environment.
Suite names are validated at config-parse time (fail before the job starts,
not 20 minutes in).
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from src.support.results import BenchmarkResult, BenchmarkSuiteResult, TaskResult


@contextmanager
def _allow_full_unpickle():
    """LIBERO's get_task_init_states() calls torch.load WITHOUT
    weights_only=False; under torch>=2.6 the weights_only=True default rejects
    the numpy globals in LIBERO's (trusted, local) init-state files. Restore
    pre-2.6 behavior around that single call only."""
    import torch

    _orig = torch.load

    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    torch.load = _patched
    try:
        yield
    finally:
        torch.load = _orig


if TYPE_CHECKING:
    from src.configs.schemas import BenchmarkConfig

log = logging.getLogger(__name__)

KNOWN_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
_ALIASES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "10": "libero_10",
    "libero10": "libero_10",
}

# Raw robosuite render key -> canonical dataset key (HuggingFaceVLA/libero
# naming, per the §6.1 contract: adapters reuse their training rename maps).
_CAMERA_KEYS = {
    "agentview_image": "image",
    "robot0_eye_in_hand_image": "image2",
}


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion -> axis-angle exponential coordinates.

    Same math as robosuite's `transform_utils.quat2axisangle` (what the
    lerobot libero wrapper uses for the state vector) — reimplemented so the
    benchmark needs no robosuite import.
    """
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return np.asarray(quat[:3], dtype=np.float64) * 2.0 * math.acos(w) / den


def _canonical_obs(raw: dict, instruction: str) -> dict:
    """Raw robosuite obs -> the canonical observation (design §6.1).

    Off-screen renders arrive 180°-rotated, so flip both axes; the 8-dim
    state composition mirrors lerobot `envs/libero.py` (the validated
    reference recipe): eef_pos(3) + axis-angle(eef_quat)(3) + gripper(2).
    """
    images = {canon: np.ascontiguousarray(raw[k][::-1, ::-1]) for k, canon in _CAMERA_KEYS.items()}
    state = np.concatenate(
        (
            raw["robot0_eef_pos"],
            _quat2axisangle(raw["robot0_eef_quat"]),
            raw["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return {"images": images, "state": state, "instruction": instruction}


class LIBEROBenchmark:
    name = "libero"
    suites = list(KNOWN_SUITES)

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self.max_steps = int((cfg.extra or {}).get("max_steps", 520))
        self.camera_hw = (cfg.extra or {}).get("camera_hw", [256, 256])
        # Dummy no-op steps after set_init_state so objects settle before
        # the policy's first frame (reference LIBERO evals use ~10).
        self.settle_steps = int((cfg.extra or {}).get("settle_steps", 10))

    @classmethod
    def from_config(cls, cfg: BenchmarkConfig) -> LIBEROBenchmark:
        for s in cfg.suites:
            if _ALIASES.get(s, s) not in KNOWN_SUITES:
                raise ValueError(f"Unknown LIBERO suite: {s!r} (known: {KNOWN_SUITES})")
        return cls(cfg)

    def canonical_suite(self, suite: str) -> str:
        """The canonical name of a suite (`spatial` -> `libero_spatial`)."""
        return _ALIASES.get(suite, suite)

    def task_names(self, suite: str) -> dict[int, str]:
        """task_id -> language description, for mapping dataset task names
        onto ids (personalized eval)."""
        from libero.libero.benchmark import get_benchmark_dict

        s = get_benchmark_dict()[_ALIASES.get(suite, suite)]()
        return {tid: s.get_task(tid).language for tid in range(s.n_tasks)}

    def _selected_task_ids(self, suite: str, n_tasks: int) -> list[int]:
        """All task ids of a suite, narrowed by cfg.extra['task_filter'] when
        set ({suite: [task_ids]} — the personalized-eval per-client subset).
        The filter is keyed by the RAW config suite name (pre-alias)."""
        task_filter = (self.cfg.extra or {}).get("task_filter")
        if task_filter is None or suite not in task_filter:
            return list(range(n_tasks))
        wanted = [t for t in task_filter[suite] if 0 <= int(t) < n_tasks]
        if len(wanted) != len(task_filter[suite]):
            raise ValueError(f"task_filter[{suite!r}]={task_filter[suite]} has ids outside " f"0..{n_tasks - 1}")
        return [int(t) for t in wanted]

    def evaluate(
        self,
        policy: Any,
        suites: list[str],
        n_episodes_per_task: int,
        seed: int,
        progress_cb: Callable[[str, int, int, bool], None] | None = None,
    ) -> BenchmarkResult:
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark_dict
        from libero.libero.envs import OffScreenRenderEnv

        suite_results: dict[str, BenchmarkSuiteResult] = {}
        total_eps = 0
        total_succ = 0

        for raw_name in suites:
            sname = _ALIASES.get(raw_name, raw_name)
            suite = get_benchmark_dict()[sname]()
            per_task: list[TaskResult] = []
            suite_eps = 0
            suite_succ = 0
            for task_id in self._selected_task_ids(raw_name, suite.n_tasks):
                n_succ = 0
                task = suite.get_task(task_id)
                with _allow_full_unpickle():
                    init_states = suite.get_task_init_states(task_id)
                # One env per task, reused across its episodes: construction
                # parses BDDL and builds the MuJoCo model (seconds each).
                # libero's task.bddl_file is a bare filename on some builds;
                # join it under the bddl_files root when it isn't a real path.
                bddl_path = task.bddl_file
                if not os.path.exists(bddl_path):
                    bddl_path = os.path.join(
                        get_libero_path("bddl_files"),
                        task.problem_folder,
                        os.path.basename(task.bddl_file),
                    )
                env = OffScreenRenderEnv(
                    bddl_file_name=bddl_path,
                    camera_heights=self.camera_hw[0],
                    camera_widths=self.camera_hw[1],
                )
                # Determinism comes primarily from the fixed init_states;
                # seeding additionally pins controller/MuJoCo noise where the
                # env supports it. Cluster-validated in V1.
                if hasattr(env, "seed"):
                    env.seed(seed)
                try:
                    for ep in range(n_episodes_per_task):
                        ok = self._rollout(policy, env, task, init_states, ep, seed)
                        n_succ += int(ok)
                        if progress_cb:
                            progress_cb(sname, task_id, ep, ok)
                finally:
                    env.close()
                per_task.append(
                    TaskResult(
                        task_id=task_id,
                        task_name=task.language,
                        n_episodes=n_episodes_per_task,
                        n_success=n_succ,
                        success_rate=n_succ / n_episodes_per_task if n_episodes_per_task else 0.0,
                    )
                )
                suite_eps += n_episodes_per_task
                suite_succ += n_succ
            suite_results[sname] = BenchmarkSuiteResult(
                n_episodes=suite_eps,
                n_success=suite_succ,
                success_rate=suite_succ / suite_eps if suite_eps else 0.0,
                per_task=per_task,
            )
            total_eps += suite_eps
            total_succ += suite_succ
            log.info(
                "LIBERO suite %s: %d/%d episodes successful",
                sname,
                suite_succ,
                suite_eps,
            )

        return BenchmarkResult(
            benchmark=self.name,
            suites=suite_results,
            overall_n_episodes=total_eps,
            overall_n_success=total_succ,
            overall_success_rate=total_succ / total_eps if total_eps else 0.0,
        )

    def _rollout(self, policy: Any, env, task, init_states, episode_idx: int, seed: int) -> bool:
        env.reset()
        obs = env.set_init_state(init_states[episode_idx % len(init_states)])
        # Settle phase: step dummy no-op actions so objects dropped by
        # set_init_state come to rest before the policy's first frame.
        # Not counted against max_steps.
        settle_action = np.array([0.0] * 6 + [-1.0])
        for _ in range(self.settle_steps):
            obs, _r, done, _info = env.step(settle_action)
            if done:
                # done while settling is degenerate — count as failure
                return False
        policy.reset({"language": task.language, "seed": seed, "episode": episode_idx})
        for _ in range(self.max_steps):
            action = np.asarray(policy.select_action(_canonical_obs(obs, task.language)))
            obs, _r, done, info = env.step(action)
            if done:
                # Real LIBERO/robosuite envs don't put "success" in info:
                # `done` itself is the success signal (timeout is the only
                # other exit, handled by the max_steps cap). Fakes/forks may
                # carry an explicit info["success"], which wins if present.
                return bool(info.get("success", done))
        return False

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

"""LIBEROBenchmark against a faked `libero` package — CPU only."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

_NOOP = np.array([0.0] * 6 + [-1.0])


def _raw_obs(eef_pos=(1.0, 2.0, 3.0)):
    """Raw robosuite-style obs as OffScreenRenderEnv returns them: flat keys,
    180°-rotated renders."""
    img = np.zeros((4, 4, 3), np.uint8)
    img[0, 0] = (9, 8, 7)  # marker top-left pre-flip
    return {
        "agentview_image": img,
        "robot0_eye_in_hand_image": np.full((4, 4, 3), 50, np.uint8),
        "robot0_eef_pos": np.asarray(eef_pos, dtype=np.float64),
        "robot0_eef_quat": np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
    }


class _FakeTask:
    def __init__(self, i):
        self.language = f"task {i}"
        self.problem_folder = "f"
        self.bddl_file = f"t{i}.bddl"


class _FakeSuite:
    n_tasks = 2

    def get_task(self, i):
        return _FakeTask(i)

    def get_task_init_states(self, i):
        return [np.zeros(2)] * 3


class _FakeEnv:
    """Real-API fake: no `seed` attr and no info["success"] — on real
    LIBERO/robosuite, `done` itself is the success signal."""

    def __init__(self, **kw):
        self.t = 0

    def reset(self):
        return _raw_obs()

    def set_init_state(self, s):
        self.t = 0
        return _raw_obs()

    def step(self, action):
        self.t += 1
        done = self.t >= 2
        return _raw_obs(), 0.0, done, {}

    def close(self):
        pass


@pytest.fixture(autouse=True)
def fake_libero(monkeypatch):
    pkg = types.ModuleType("libero")
    lib = types.ModuleType("libero.libero")
    bench = types.ModuleType("libero.libero.benchmark")
    envs = types.ModuleType("libero.libero.envs")
    bench.get_benchmark_dict = lambda: {"libero_spatial": _FakeSuite}
    envs.OffScreenRenderEnv = _FakeEnv
    # benchmark.py resolves a task's bddl path via get_libero_path when the
    # bare bddl_file name isn't a real path (the case for these CPU fakes).
    lib.get_libero_path = lambda key: f"/fake/libero/{key}"
    lib.benchmark, lib.envs = bench, envs
    pkg.libero = lib
    for n, m in [
        ("libero", pkg),
        ("libero.libero", lib),
        ("libero.libero.benchmark", bench),
        ("libero.libero.envs", envs),
    ]:
        monkeypatch.setitem(sys.modules, n, m)
    for m in list(sys.modules):
        if m.startswith("decent_vla.benchmarks.libero"):
            monkeypatch.delitem(sys.modules, m, raising=False)
    yield types.SimpleNamespace(bench=bench, envs=envs)


class _MockPolicy:
    action_horizon = 1

    def reset(self, task_meta):
        self.task = task_meta

    def select_action(self, observation):
        return np.zeros(7)


def _make(extra):
    from decent_vla.benchmarks.libero import LIBEROBenchmark
    from decent_vla.configs.schemas import BenchmarkConfig

    return LIBEROBenchmark.from_config(BenchmarkConfig(name="libero", suites=["libero_spatial"], extra=extra))


def test_libero_benchmark_evaluates():
    # _FakeEnv returns empty info: done alone must count as success
    # (the real LIBERO/robosuite convention).
    b = _make({"max_steps": 5, "settle_steps": 0})
    res = b.evaluate(_MockPolicy(), suites=["libero_spatial"], n_episodes_per_task=2, seed=0)
    s = res.suites["libero_spatial"]
    assert s.n_episodes == 4 and s.n_success == 4
    assert s.success_rate == 1.0
    assert len(s.per_task) == 2
    assert [t.task_id for t in s.per_task] == [0, 1]
    assert s.per_task[0].task_name == "task 0"
    assert s.per_task[0].n_episodes == 2 and s.per_task[0].n_success == 2
    assert res.benchmark == "libero"
    assert res.overall_n_episodes == 4
    assert res.overall_n_success == 4
    assert res.overall_success_rate == 1.0


def test_explicit_info_success_false_wins(fake_libero):
    """A fake/fork that *does* report info["success"]=False on done must
    override the done==success default."""

    class _ExplicitFailEnv(_FakeEnv):
        def step(self, action):
            self.t += 1
            done = self.t >= 2
            return _raw_obs(), 0.0, done, {"success": False} if done else {}

    fake_libero.envs.OffScreenRenderEnv = _ExplicitFailEnv
    b = _make({"max_steps": 5, "settle_steps": 0})
    res = b.evaluate(_MockPolicy(), suites=["libero_spatial"], n_episodes_per_task=2, seed=0)
    assert res.suites["libero_spatial"].n_success == 0


def test_env_constructed_once_per_task(fake_libero):
    """Env construction is expensive (BDDL parse + MuJoCo build): exactly
    one env per task, reused across that task's episodes, then closed."""
    counts = {"init": 0, "close": 0}

    class _CountingEnv(_FakeEnv):
        def __init__(self, **kw):
            counts["init"] += 1
            super().__init__(**kw)

        def close(self):
            counts["close"] += 1

    fake_libero.envs.OffScreenRenderEnv = _CountingEnv
    b = _make({"max_steps": 5, "settle_steps": 0})
    b.evaluate(_MockPolicy(), suites=["libero_spatial"], n_episodes_per_task=2, seed=0)
    assert counts["init"] == 2  # 2 tasks x 2 episodes -> 2 envs, not 4
    assert counts["close"] == 2


def test_env_seeded_when_supported(fake_libero):
    seeds = []

    class _SeededEnv(_FakeEnv):
        def seed(self, s):
            seeds.append(s)

    fake_libero.envs.OffScreenRenderEnv = _SeededEnv
    b = _make({"max_steps": 5, "settle_steps": 0})
    b.evaluate(_MockPolicy(), suites=["libero_spatial"], n_episodes_per_task=1, seed=42)
    assert seeds == [42, 42]  # once per task env


def test_settle_steps_precede_policy_control(fake_libero):
    """After set_init_state the env takes `settle_steps` no-op actions that
    do not count against max_steps; the policy's first obs is post-settle."""
    instances = []

    class _RecordingEnv:
        def __init__(self, **kw):
            self.t = 0
            self.actions = []
            instances.append(self)

        def reset(self):
            return _raw_obs(eef_pos=(self.t, 0.0, 0.0))

        def set_init_state(self, s):
            return _raw_obs(eef_pos=(self.t, 0.0, 0.0))

        def step(self, action):
            self.actions.append(np.asarray(action, dtype=float))
            self.t += 1
            return _raw_obs(eef_pos=(self.t, 0.0, 0.0)), 0.0, False, {}

        def close(self):
            pass

    class _OneTaskSuite(_FakeSuite):
        n_tasks = 1

    class _ObsRecordingPolicy(_MockPolicy):
        def __init__(self):
            self.seen = []

        def select_action(self, observation):
            self.seen.append(observation)
            return np.zeros(7)

    fake_libero.envs.OffScreenRenderEnv = _RecordingEnv
    fake_libero.bench.get_benchmark_dict = lambda: {"libero_spatial": _OneTaskSuite}
    policy = _ObsRecordingPolicy()
    b = _make({"max_steps": 4, "settle_steps": 3})
    b.evaluate(policy, suites=["libero_spatial"], n_episodes_per_task=1, seed=0)

    (env,) = instances
    # settle steps + policy steps, nothing more
    assert len(env.actions) == 3 + 4
    for a in env.actions[:3]:
        np.testing.assert_array_equal(a, _NOOP)
    # policy's first frame is the post-settle obs (env stamps t into eef_pos)
    assert policy.seen[0]["state"][0] == 3
    assert len(policy.seen) == 4


def test_done_during_settling_is_failure(fake_libero):
    """Env signalling done while settling is degenerate: not a success, and
    the policy is never asked for an action."""

    class _NeverCalledPolicy(_MockPolicy):
        def select_action(self, observation):
            raise AssertionError("policy must not act during settling")

    # _FakeEnv goes done at t>=2 with empty info; settle_steps=5 > 2.
    b = _make({"max_steps": 5, "settle_steps": 5})
    res = b.evaluate(_NeverCalledPolicy(), suites=["libero_spatial"], n_episodes_per_task=2, seed=0)
    assert res.suites["libero_spatial"].n_success == 0


def test_unknown_suite_fails_at_config():
    from decent_vla.benchmarks.libero import LIBEROBenchmark
    from decent_vla.configs.schemas import BenchmarkConfig

    with pytest.raises(ValueError, match="Unknown LIBERO suite"):
        LIBEROBenchmark.from_config(BenchmarkConfig(name="libero", suites=["bogus"]))


def test_progress_cb_called_per_episode():
    calls = []
    b = _make({"max_steps": 5, "settle_steps": 0})
    b.evaluate(
        _MockPolicy(),
        suites=["libero_spatial"],
        n_episodes_per_task=2,
        seed=0,
        progress_cb=lambda suite, task_id, episode, success: calls.append((suite, task_id, episode, success)),
    )
    assert calls == [
        ("libero_spatial", 0, 0, True),
        ("libero_spatial", 0, 1, True),
        ("libero_spatial", 1, 0, True),
        ("libero_spatial", 1, 1, True),
    ]


def test_rollout_passes_canonical_observation():
    """§6.1 contract: the benchmark hands the policy upright uint8 images
    under dataset-convention names, the 8-dim lerobot-reference state, and
    the task language — never raw robosuite keys."""

    class _Recorder(_MockPolicy):
        def __init__(self):
            self.seen = []

        def select_action(self, observation):
            self.seen.append(observation)
            return np.zeros(7)

    policy = _Recorder()
    b = _make({"max_steps": 1, "settle_steps": 0})
    b.evaluate(policy, suites=["libero_spatial"], n_episodes_per_task=1, seed=0)

    obs = policy.seen[0]
    assert set(obs) == {"images", "state", "instruction"}
    assert set(obs["images"]) == {"image", "image2"}
    img = obs["images"]["image"]
    assert img.dtype == np.uint8
    # 180° flip: pre-flip top-left marker lands bottom-right
    np.testing.assert_array_equal(img[-1, -1], (9, 8, 7))
    assert not img[0, 0].any()
    expected_state = np.concatenate(([1.0, 2.0, 3.0], [0.0, 0.0, np.pi / 2], [0.04, -0.04]))
    assert obs["state"].shape == (8,)
    np.testing.assert_allclose(obs["state"], expected_state, atol=1e-6)
    assert obs["instruction"] == "task 0"


def test_quat2axisangle_matches_robosuite_reference():
    from decent_vla.benchmarks.libero.benchmark import _quat2axisangle

    # 90° about z
    np.testing.assert_allclose(
        _quat2axisangle(np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])),
        [0.0, 0.0, np.pi / 2],
        atol=1e-7,
    )
    # identity rotation -> zero vector
    np.testing.assert_array_equal(_quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0])), np.zeros(3))


def test_suite_alias_resolves_to_canonical_key():
    from decent_vla.benchmarks.libero import LIBEROBenchmark
    from decent_vla.configs.schemas import BenchmarkConfig

    b = LIBEROBenchmark.from_config(
        BenchmarkConfig(name="libero", suites=["spatial"], extra={"max_steps": 5, "settle_steps": 0})
    )
    res = b.evaluate(_MockPolicy(), suites=["spatial"], n_episodes_per_task=1, seed=0)
    assert "libero_spatial" in res.suites

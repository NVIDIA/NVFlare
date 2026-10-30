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

"""`eval.n_action_steps` — the action horizon used at rollout time.

Chunked VLA policies (SmolVLA, pi0.5, ...) predict `chunk_size` actions at once
and replan only when the queue empties, so `config.n_action_steps` IS the
replanning interval. Both SmolVLA and pi0.5 default it to 50 = the full chunk,
i.e. fully open loop: the policy sees the world once every 50 environment steps.

decent-vla's own LIBERO benchmark inherited that default while `lerobot-eval`
passes `--policy.n_action_steps=10`. On one identical checkpoint
(smolvla_libero_fl_noniid_fullft_unfrozen round 110, 400 episodes) that was
worth **33 points**: 70.0% at nact=10 vs 37.0% at nact=50, and the gap was not
uniform across suites (-19 to -47). decent-vla's harness therefore ran a
different protocol from `lerobot-eval`.
"""

from __future__ import annotations

import pytest

from decent_vla.configs.schemas import EvalConfig


def test_eval_config_carries_an_action_horizon():
    """It must be settable from the run config, and default to "leave alone"."""
    assert EvalConfig().n_action_steps is None, "default must not silently change any policy"
    assert EvalConfig(n_action_steps=10).n_action_steps == 10


class _Cfg:
    def __init__(self, n):
        self.n_action_steps = n


class _Policy:
    """Stands in for a lerobot policy: an object carrying a `.config`."""

    def __init__(self, n):
        self.config = _Cfg(n)


def test_apply_action_horizon_sets_it_and_reports_the_change():
    from decent_vla.runtime.eval_runner import apply_action_horizon

    p = _Policy(50)
    changed = apply_action_horizon(p, 10)
    assert p.config.n_action_steps == 10
    assert changed == (50, 10)


def test_apply_action_horizon_is_a_noop_when_unset():
    from decent_vla.runtime.eval_runner import apply_action_horizon

    p = _Policy(50)
    assert apply_action_horizon(p, None) is None
    assert p.config.n_action_steps == 50, "None must leave the policy untouched"


def test_apply_action_horizon_is_a_noop_when_already_correct():
    from decent_vla.runtime.eval_runner import apply_action_horizon

    p = _Policy(10)
    assert apply_action_horizon(p, 10) is None
    assert p.config.n_action_steps == 10


def test_apply_action_horizon_raises_on_a_policy_that_cannot_take_one():
    """Silently ignoring the request would reproduce the original bug: the eval
    would run at a horizon the config did not ask for and nothing would say so."""
    from decent_vla.runtime.eval_runner import apply_action_horizon

    class _NoConfig:
        pass

    with pytest.raises(ValueError, match="n_action_steps"):
        apply_action_horizon(_NoConfig(), 10)


def test_action_horizon_is_part_of_the_eval_cache_provenance_key(tmp_path):
    """Changing the horizon must invalidate cached suites.

    Both eval paths cache per-suite/per-client results keyed on a provenance
    dict and skip anything whose key matches. The key was
    {checkpoint, seed, n_episodes_per_task} — none of which change when the
    action horizon does. Re-running an arm at nact=10 would therefore have been
    served the cached nact=50 numbers, silently, which is the exact trap this
    whole fix exists to close.

    eval_runner folds the whole eval config into `eval_config_hash`, so the
    horizon is covered iff that hash moves when the horizon does — asserted
    behaviorally. personalized_eval still lists the horizon explicitly.
    """
    import inspect

    from decent_vla.runtime import eval_runner, personalized_eval
    from decent_vla.runtime.eval_runner import _eval_config_hash

    h10 = _eval_config_hash(_load(tmp_path, "  n_action_steps: 10\n"))
    h50 = _eval_config_hash(_load(tmp_path, "  n_action_steps: 50\n"))
    assert h10 != h50, "eval_config_hash must change when n_action_steps does"

    src = inspect.getsource(eval_runner)
    start = src.index("provenance = {")
    block = src[start : src.index("}", start)]
    assert "eval_config_hash" in block, f"eval_runner provenance key omits eval_config_hash: {block!r}"

    src = inspect.getsource(personalized_eval)
    start = src.index("provenance = {")
    block = src[start : src.index("}", start)]
    assert "n_action_steps" in block, f"personalized_eval provenance key omits n_action_steps: {block!r}"


# ---------------------------------------------------------------------------
# The hop that was missing. Everything above passed while the feature was inert.
# ---------------------------------------------------------------------------


def _load(tmp_path, eval_block: str):
    from decent_vla.configs.loader import load_run_config

    p = tmp_path / "run.yaml"
    p.write_text("name: probe\nbackend: {name: mock}\neval:\n  only: true\n" "  benchmark: {name: mock}\n" + eval_block)
    return load_run_config(str(p))


def test_action_horizon_survives_the_yaml_to_config_hop(tmp_path):
    """The gap this whole module was written for, and missed.

    `EvalConfig` had the field, `apply_action_horizon` honored it, the cache key
    recorded it, the YAMLs set it — but `loader.py::_build_eval` never read it,
    so every value arrived as None and `apply_action_horizon` returned at its
    first line. Months of evals ran fully open loop in directories named
    `_nact10`. Every test above passes with the loader still broken; only a
    round trip from real YAML catches it.
    """
    assert _load(tmp_path, "  n_action_steps: 10\n").eval.n_action_steps == 10


def test_action_horizon_defaults_to_none_when_the_yaml_omits_it(tmp_path):
    assert _load(tmp_path, "").eval.n_action_steps is None


# ---------------------------------------------------------------------------
# Unwrapping. The guard above was right to refuse, but it was looking in the
# wrong place, so the horizon still never reached a real eval.
# ---------------------------------------------------------------------------


class _Wrapper:
    """Shaped like adapters/lerobot/policy.py::LeRobotPolicy — the object the
    benchmark actually drives. It holds the lerobot policy in `_policy` and
    exposes no `.config` of its own."""

    def __init__(self, inner):
        self._policy = inner


def test_horizon_reaches_a_policy_behind_the_adapter_wrapper():
    from decent_vla.runtime.eval_runner import apply_action_horizon

    """What ran on the GPU: build_policy() returns the wrapper, so
    `policy.config` is absent and the guard raised
    `LeRobotPolicy exposes no integer config.n_action_steps` on every pi0.5
    eval — after loading a 3.2B model. The horizon has to be found through
    the wrapper."""
    wrapped = _Wrapper(_Policy(50))
    assert apply_action_horizon(wrapped, 10) == (50, 10)
    assert wrapped._policy.config.n_action_steps == 10


def test_a_wrapper_around_a_policy_without_a_horizon_still_raises():
    from decent_vla.runtime.eval_runner import apply_action_horizon

    """GR00T must keep failing loudly rather than running at some other
    horizon — wrapping must not turn the guard off."""

    class _NoCfg:
        pass

    with pytest.raises(ValueError, match="no integer config.n_action_steps"):
        apply_action_horizon(_Wrapper(_NoCfg()), 10)

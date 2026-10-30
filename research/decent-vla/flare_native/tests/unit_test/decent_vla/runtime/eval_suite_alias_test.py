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

"""run_eval accepts a benchmark that keys a suite's result by its canonical name (LIBERO aliases)."""

from __future__ import annotations

import dataclasses
import json

from decent_vla.benchmarks._mock import MockBenchmark
from decent_vla.cli.main import main
from tests.eval_test_utils import _write_train_eval_cfg


class _CanonicalKeyBenchmark:
    """Mock benchmark that renames each suite in its result, as LIBERO does for `spatial`."""

    def __init__(self, inner):
        self._inner = inner
        self.name = inner.name

    def canonical_suite(self, suite):
        return suite if suite.startswith("canonical_") else f"canonical_{suite}"

    def evaluate(self, policy, suites, n_episodes_per_task, seed, **kw):
        inner_suites = [s.removeprefix("canonical_") for s in suites]
        result = self._inner.evaluate(policy, inner_suites, n_episodes_per_task, seed, **kw)
        renamed = {f"canonical_{k}": v for k, v in result.suites.items()}
        return dataclasses.replace(result, suites=renamed)


def test_eval_takes_the_result_of_an_aliased_suite(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "decent_vla.runtime.eval_runner.build_benchmark",
        lambda eval_cfg: _CanonicalKeyBenchmark(MockBenchmark.from_config(eval_cfg.benchmark)),
    )
    out = tmp_path / "out"
    cfg_path = _write_train_eval_cfg(tmp_path, "c.yaml", num_rounds=1, output_dir=out)
    assert main(["eval", "--config", str(cfg_path)]) == 0
    data = json.loads((out / "eval" / "results.json").read_text())
    # Results and the per-suite cache use the canonical name, which held-out specs use too.
    assert list(data["suites"]) == ["canonical_default"]
    assert (out / "eval" / "suites" / "canonical_default.json").exists()
    assert data["overall"]["n_episodes"] > 0


def test_an_alias_and_its_canonical_name_do_not_share_a_cached_suite(tmp_path, monkeypatch):
    """An alias selects tasks by its own name, so switching a config from the alias to the
    canonical name must re-evaluate rather than reuse the alias's cached result."""

    def build(eval_cfg):
        inner = dataclasses.replace(
            eval_cfg.benchmark, suites=[s.removeprefix("canonical_") for s in eval_cfg.benchmark.suites]
        )
        return _CanonicalKeyBenchmark(MockBenchmark.from_config(inner))

    monkeypatch.setattr("decent_vla.runtime.eval_runner.build_benchmark", build)
    out = tmp_path / "out"
    alias_cfg = _write_train_eval_cfg(tmp_path, "a.yaml", num_rounds=1, output_dir=out, suites=["default"])
    assert main(["eval", "--config", str(alias_cfg)]) == 0
    cached = out / "eval" / "suites" / "canonical_default.json"
    first = json.loads(cached.read_text())["provenance"]
    # The same alias config again hits its own cache: the suite file is not rewritten.
    mtime = cached.stat().st_mtime_ns
    assert main(["eval", "--config", str(alias_cfg)]) == 0
    assert cached.stat().st_mtime_ns == mtime
    canon_cfg = _write_train_eval_cfg(tmp_path, "c.yaml", num_rounds=1, output_dir=out, suites=["canonical_default"])
    assert main(["eval", "--config", str(canon_cfg)]) == 0
    second = json.loads(cached.read_text())["provenance"]
    assert first.get("config_suite") == "default"
    assert "config_suite" not in second


def test_a_suite_listed_by_alias_and_full_name_is_rejected(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(
        "decent_vla.runtime.eval_runner.build_benchmark",
        lambda eval_cfg: _CanonicalKeyBenchmark(
            MockBenchmark.from_config(dataclasses.replace(eval_cfg.benchmark, suites=["default"]))
        ),
    )
    cfg = _write_train_eval_cfg(
        tmp_path, "d.yaml", num_rounds=1, output_dir=tmp_path / "out", suites=["default", "canonical_default"]
    )
    with pytest.raises(ValueError, match="same suite twice"):
        main(["eval", "--config", str(cfg)])

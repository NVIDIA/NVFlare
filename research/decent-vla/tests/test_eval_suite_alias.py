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
from tests.test_cli import _write_train_eval_cfg


class _CanonicalKeyBenchmark:
    """Mock benchmark that renames each suite in its result, as LIBERO does for `spatial`."""

    def __init__(self, inner):
        self._inner = inner
        self.name = inner.name

    def canonical_suite(self, suite):
        return f"canonical_{suite}"

    def evaluate(self, policy, suites, n_episodes_per_task, seed, **kw):
        result = self._inner.evaluate(policy, suites, n_episodes_per_task, seed, **kw)
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

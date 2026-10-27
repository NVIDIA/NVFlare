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

"""The eval cache key covers the lerobot pipeline YAML that the policy is built from."""
from __future__ import annotations

from decent_vla.configs.schemas import BackendConfig, EvalConfig, RunConfig
from decent_vla.runtime.eval_runner import _eval_config_hash


def _cfg(extra):
    return RunConfig(name="t", backend=BackendConfig(name="lerobot", extra=extra), eval=EvalConfig())


def test_editing_the_train_config_file_changes_the_key(tmp_path):
    tp = tmp_path / "pipeline.yaml"
    tp.write_text("rename_map: {a: b}\n")
    cfg = _cfg({"train_config_path": str(tp)})
    before = _eval_config_hash(cfg)
    tp.write_text("rename_map: {a: c}\n")
    assert _eval_config_hash(cfg) != before


def test_key_without_a_train_config_path_is_unchanged_by_the_new_field():
    import dataclasses
    import json

    from decent_vla.core.results import config_hash
    from decent_vla.runtime.eval_runner import _EVAL_CACHE_IGNORED

    cfg = _cfg({})
    ev = {k: v for k, v in dataclasses.asdict(cfg.eval).items() if k not in _EVAL_CACHE_IGNORED}
    ev["benchmark"] = {k: v for k, v in ev["benchmark"].items() if k != "suites"}
    legacy = config_hash(json.dumps({"backend": dataclasses.asdict(cfg.backend), "eval": ev},
                                    sort_keys=True, separators=(",", ":"), default=str))
    assert _eval_config_hash(cfg) == legacy

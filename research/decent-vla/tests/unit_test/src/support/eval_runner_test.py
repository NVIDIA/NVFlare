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

"""The evaluation cache key, which must follow the repositories that set a policy's normalization statistics."""

from __future__ import annotations

import dataclasses
import json

from src.config import BackendConfig, DataConfig, EvalConfig, RunConfig
from src.support.eval_runner import _eval_config_hash
from src.support.results import config_hash


def _cfg(client_repo_ids=None):
    return RunConfig(
        name="run",
        backend=BackendConfig(name="lerobot"),
        data=DataConfig(client_repo_ids=client_repo_ids),
        eval=EvalConfig(),
    )


def test_client_repo_ids_change_the_key():
    assignments = (None, [["org/a"]], [["org/b"]], [["org/a"], ["org/b"]])

    assert len({_eval_config_hash(_cfg(assignment)) for assignment in assignments}) == len(assignments)


def test_same_client_repo_ids_give_the_same_key():
    assert _eval_config_hash(_cfg([["org/a"]])) == _eval_config_hash(_cfg([["org/a"]]))


def test_runs_without_client_repo_ids_keep_their_key():
    # The payload written before client_repo_ids joined the key, so that existing caches stay valid.
    cfg = _cfg()
    evaluation = dataclasses.asdict(cfg.eval)
    del evaluation["output_dir"], evaluation["personalized"], evaluation["benchmark"]["suites"]
    key = {"backend": dataclasses.asdict(cfg.backend), "eval": evaluation}
    payload = json.dumps(key, sort_keys=True, separators=(",", ":"), default=str)

    assert _eval_config_hash(cfg) == config_hash(payload)

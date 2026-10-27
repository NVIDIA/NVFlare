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

"""explicit_repos config surface — no lerobot needed."""
from __future__ import annotations

import textwrap

from decent_vla.configs import load_run_config
from decent_vla.configs.schemas import DataConfig


def _write(tmp_path, body: str):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(body))
    return str(p)


def test_dataconfig_has_client_repo_ids_default_none():
    assert DataConfig().client_repo_ids is None


def test_explicit_repos_parses_client_repo_ids(tmp_path):
    cfg = load_run_config(_write(tmp_path, """
        name: t
        backend: {name: mock}
        data:
          name: lerobot
          partitioning: {scheme: explicit_repos}
          client_repo_ids: [["a/x", "a/y"], ["b/z"]]
        fl: {strategy: {name: fedavg}, rounds: 1, local_steps: 1, min_clients: 2}
    """))
    assert cfg.data.partitioning == "explicit_repos"
    assert cfg.data.client_repo_ids == [["a/x", "a/y"], ["b/z"]]


def test_explicit_repos_string_scheme_form(tmp_path):
    cfg = load_run_config(_write(tmp_path, """
        name: t
        backend: {name: mock}
        data:
          name: lerobot
          partitioning: explicit_repos
          client_repo_ids: [["a/x"], ["b/z"]]
        fl: {strategy: {name: fedavg}, rounds: 1, local_steps: 1, min_clients: 2}
    """))
    assert cfg.data.partitioning == "explicit_repos"
    assert len(cfg.data.client_repo_ids) == 2

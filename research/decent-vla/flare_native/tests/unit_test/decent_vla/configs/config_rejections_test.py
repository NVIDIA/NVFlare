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

"""Settings the code accepts in YAML but does not apply are rejected at load time, not run silently."""

from __future__ import annotations

import textwrap

import pytest

from decent_vla.configs import load_run_config
from decent_vla.configs.loader import ConfigError


def _load(tmp_path, body: str):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(body))
    return load_run_config(str(p))


def test_aggregate_lora_only_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="aggregate_lora_only"):
        _load(
            tmp_path,
            """
            name: t
            backend: {name: mock, lora: {r: 4}}
            fl: {strategy: {name: fedavg, aggregate_lora_only: true}, rounds: 1, local_steps: 1}
        """,
        )


@pytest.mark.parametrize("keep", [0, 1])
def test_client_state_keep_below_two_is_rejected(tmp_path, keep):
    with pytest.raises(ConfigError, match="client_state_keep"):
        _load(
            tmp_path,
            f"""
            name: t
            backend: {{name: mock}}
            fl: {{strategy: {{name: fedavg}}, rounds: 1, local_steps: 1, client_state_keep: {keep}}}
        """,
        )


def test_client_state_keep_of_two_is_accepted(tmp_path):
    cfg = _load(
        tmp_path,
        """
        name: t
        backend: {name: mock}
        fl: {strategy: {name: fedavg}, rounds: 1, local_steps: 1, client_state_keep: 2}
    """,
    )
    assert cfg.fl.client_state_keep == 2


def test_lerobot_lora_attach_to_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="attach_to"):
        _load(
            tmp_path,
            """
            name: t
            backend: {name: lerobot, lora: {r: 4, attach_to: model.vlm}}
            fl: {strategy: {name: fedavg}, rounds: 1, local_steps: 1}
        """,
        )


def test_data_revision_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="data.revision"):
        _load(
            tmp_path,
            """
            name: t
            backend: {name: mock}
            data: {revision: v2.1}
            fl: {strategy: {name: fedavg}, rounds: 1, local_steps: 1}
        """,
        )


def test_explicit_repos_derives_num_clients_from_the_repo_lists(tmp_path):
    cfg = _load(
        tmp_path,
        """
        name: t
        backend: {name: mock}
        data: {name: lerobot, partitioning: explicit_repos, client_repo_ids: [["a/x"], ["b/y"], ["c/z"]]}
        fl: {strategy: {name: fedavg}, rounds: 1, local_steps: 1}
    """,
    )
    assert cfg.data.num_clients == 3


def test_explicit_repos_count_mismatch_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="one list per client"):
        _load(
            tmp_path,
            """
            name: t
            backend: {name: mock}
            data: {name: lerobot, partitioning: explicit_repos, num_clients: 3, client_repo_ids: [["a/x"], ["b/y"]]}
            fl: {strategy: {name: fedavg}, rounds: 1, local_steps: 1}
        """,
        )

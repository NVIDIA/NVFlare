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

"""The LeRobot adapter's check that episodes are partitioned from the same dataset training loads.

No LeRobot, model or data: each test controls whether a local dataset copy exists through a stand-in HF_LEROBOT_HOME,
and from_config stops where it would go on to load the dataset metadata, before any policy is built.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from src.adapters.lerobot.backend import LeRobotBackend
from src.config import BackendConfig, DataConfig

REPO_ID = "org/dataset"


class _PastIdentityCheck(Exception):
    """Raised where from_config goes on to load the dataset metadata."""


@pytest.fixture(autouse=True)
def stop_after_identity_check(monkeypatch):
    def load_metadata(tp_cfg):
        raise _PastIdentityCheck

    monkeypatch.setattr("src.adapters.lerobot.backend._get_dataset_meta", load_metadata)


@pytest.fixture(autouse=True)
def lerobot_home(monkeypatch, tmp_path):
    """An HF_LEROBOT_HOME holding no dataset, so an unset root means the Hugging Face Hub copy."""
    home = tmp_path / "lerobot_home"
    monkeypatch.setitem(sys.modules, "lerobot.utils.constants", SimpleNamespace(HF_LEROBOT_HOME=home))
    return home


def _data(repo_id=REPO_ID, root=None, **kwargs):
    return DataConfig(repo_id=repo_id, extra={"root": root} if root else {}, **kwargs)


def _from_config(data, train_repo_id=REPO_ID, train_root=None):
    train_pipeline = SimpleNamespace(dataset=SimpleNamespace(repo_id=train_repo_id, root=train_root))
    cfg = BackendConfig(name="lerobot", extra={"train_pipeline_config": train_pipeline})
    LeRobotBackend.from_config(cfg, torch.device("cpu"), data)


def test_different_repo_id_is_rejected():
    with pytest.raises(ValueError, match="differs from dataset.repo_id"):
        _from_config(_data(), train_repo_id="org/other")


@pytest.mark.parametrize(
    "data_root, train_root",
    [("/data/a", "/data/b"), ("/data/a", None), (None, "/data/a")],
    ids=["different roots", "only data.extra.root", "only dataset.root"],
)
def test_different_root_is_rejected(data_root, train_root):
    with pytest.raises(ValueError, match="Set both to the same dataset root"):
        _from_config(_data(root=data_root), train_root=train_root)


@pytest.mark.parametrize(
    "data_root, train_root",
    [(None, None), ("/data/a", "/data/a")],
    ids=["no roots", "same root"],
)
def test_same_dataset_is_accepted(data_root, train_root):
    with pytest.raises(_PastIdentityCheck):
        _from_config(_data(root=data_root), train_root=train_root)


def test_same_root_spelled_differently_is_accepted(tmp_path):
    with pytest.raises(_PastIdentityCheck):
        _from_config(_data(root=str(tmp_path / "a")), train_root=str(tmp_path / "b" / ".." / "a"))


def test_local_copy_stands_in_for_an_unset_root(lerobot_home):
    local_copy = lerobot_home / REPO_ID
    (local_copy / "meta").mkdir(parents=True)

    # What the generated recipes do: the train pipeline YAML defaults to the local copy, data.extra.root stays unset.
    with pytest.raises(_PastIdentityCheck):
        _from_config(_data(), train_root=str(local_copy))
    with pytest.raises(ValueError, match="Set both to the same dataset root"):
        _from_config(_data(), train_root="/data/elsewhere")


def test_explicit_repos_partitions_no_episodes_and_is_exempt():
    with pytest.raises(_PastIdentityCheck):
        _from_config(_data(repo_id="org/other", partitioning="explicit_repos"))

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

import hashlib
import pickle
import shutil
import tarfile
from pathlib import Path
from shlex import split
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from tests.integration_test.tools import prepare_cifar10


@pytest.fixture
def cifar_archive(tmp_path, monkeypatch):
    datasets = pytest.importorskip("torchvision.datasets")
    cifar = datasets.CIFAR10
    data_dir = tmp_path / cifar.base_folder
    data_dir.mkdir()
    batch = data_dir / "batch"
    batch.write_bytes(pickle.dumps({"data": np.zeros((1, 3072), dtype=np.uint8), "labels": [0]}))
    meta = data_dir / "meta"
    meta.write_bytes(pickle.dumps({"label_names": ["test"]}))
    monkeypatch.setattr(cifar, "train_list", [[batch.name, hashlib.md5(batch.read_bytes()).hexdigest()]])
    monkeypatch.setattr(cifar, "test_list", cifar.train_list)
    monkeypatch.setattr(
        cifar, "meta", {"filename": meta.name, "key": "label_names", "md5": hashlib.md5(meta.read_bytes()).hexdigest()}
    )
    archive = tmp_path / cifar.filename
    with tarfile.open(archive, "w:gz") as f:
        f.add(data_dir, arcname=cifar.base_folder)
    monkeypatch.setattr(cifar, "tgz_md5", hashlib.md5(archive.read_bytes()).hexdigest())
    monkeypatch.setattr(prepare_cifar10, "CACHE_ARCHIVE", archive)
    network = Mock(side_effect=AssertionError("Network download attempted"))
    monkeypatch.setattr(datasets.utils, "_is_remote_location_available", lambda: False)
    monkeypatch.setattr(datasets.utils, "_get_redirect_url", network)
    return cifar, archive, network


def test_cache_survives_test_teardown(cifar_archive, tmp_path, monkeypatch):
    cifar, archive, network = cifar_archive
    archive_bytes = archive.read_bytes()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = tmp_path / "home" / "data"
    config_path = Path(__file__).parents[1] / "data/test_configs/standalone_job/pt_job.yml"
    config = yaml.safe_load(config_path.read_text())

    for _ in range(2):
        for command in config["tests"][0]["setup"]:
            args = split(command)
            assert args[:3] == ["python", "tools/prepare_cifar10.py", "--root"]
            prepare_cifar10.prepare_cifar10(args[3])
        for site in range(1, config["n_clients"] + 1):
            client_root = str(root / f"site-{site}")
            assert len(cifar(root=client_root, train=True, download=False)) == 1
            assert len(cifar(root=client_root, train=False, download=False)) == 1
        shutil.rmtree(root)

    network.assert_not_called()
    assert archive.read_bytes() == archive_bytes


def test_tracking_fixture_uses_cache(cifar_archive, tmp_path_factory):
    from tests.integration_test.slow.experiment_tracking_recipes_test import cifar10_data_root

    cifar, archive, network = cifar_archive
    root = cifar10_data_root.__wrapped__(tmp_path_factory)

    for train in (True, False):
        assert len(cifar(root=str(root), train=train, download=False)) == 1
    network.assert_not_called()
    assert archive.is_file()


@pytest.mark.parametrize("cache_state", ["missing", "corrupt"])
def test_unusable_cache_preserves_torchvision_download(cifar_archive, tmp_path, cache_state):
    cifar, archive, network = cifar_archive
    if cache_state == "missing":
        archive.unlink()
    else:
        archive.write_bytes(b"corrupt archive")

    with pytest.raises(AssertionError, match="Network download attempted"):
        prepare_cifar10.prepare_cifar10(tmp_path / "data")

    assert network.call_args.args[0] == cifar.url

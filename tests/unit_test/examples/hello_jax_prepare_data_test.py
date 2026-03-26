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

import importlib.util
import os

import pytest


def _load_prepare_data_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    module_path = os.path.join(repo_root, "examples", "hello-world", "hello-jax", "prepare_data.py")
    spec = importlib.util.spec_from_file_location("hello_jax_prepare_data", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_download_if_missing_downloads_to_temp_path_then_renames(monkeypatch, tmp_path):
    module = _load_prepare_data_module()
    data_dir = tmp_path / "raw"
    filename = "train-images-idx3-ubyte.gz"
    file_path = data_dir / filename
    calls = {}

    def fake_urlretrieve(url, download_path):
        calls["url"] = url
        calls["download_path"] = download_path
        assert download_path != str(file_path)
        with open(download_path, "wb") as f:
            f.write(b"mnist-bytes")

    monkeypatch.setattr(module.urllib.request, "urlretrieve", fake_urlretrieve)

    result = module._download_if_missing(str(data_dir), filename)

    assert result == str(file_path)
    assert file_path.read_bytes() == b"mnist-bytes"
    assert calls["url"].endswith(filename)
    assert not any(path.name.endswith(".part") for path in data_dir.iterdir())


def test_download_if_missing_removes_partial_file_after_failure(monkeypatch, tmp_path):
    module = _load_prepare_data_module()
    data_dir = tmp_path / "raw"
    filename = "train-labels-idx1-ubyte.gz"
    file_path = data_dir / filename

    def fake_urlretrieve(url, download_path):
        with open(download_path, "wb") as f:
            f.write(b"partial")
        raise RuntimeError("network error")

    monkeypatch.setattr(module.urllib.request, "urlretrieve", fake_urlretrieve)

    with pytest.raises(RuntimeError, match="network error"):
        module._download_if_missing(str(data_dir), filename)

    assert not file_path.exists()
    assert not any(path.name.endswith(".part") for path in data_dir.iterdir())

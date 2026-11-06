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

import json

import pytest

from nvflare.app_common.fdn.catalog.server import fileio
from nvflare.app_common.fdn.catalog.server.fileio import write_atomically


def test_replaces_the_file_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "state.json"
    write_atomically(str(path), {"a": 1})
    write_atomically(str(path), {"a": 2})
    assert json.loads(path.read_text()) == {"a": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


@pytest.mark.parametrize("failure", ["serialize", "fsync", "replace"])
def test_failed_write_preserves_the_previous_file_and_removes_private_temp_data(tmp_path, monkeypatch, failure):
    path = tmp_path / "catalog_context.json"
    write_atomically(str(path), {"key": "previous"})
    data = {"key": "private", "previous": {"site": ["hidden"]}}
    if failure == "serialize":
        data["invalid"] = object()
        error = TypeError
    else:

        def fail(*args, **kwargs):
            raise OSError("simulated write failure")

        monkeypatch.setattr(fileio.os, failure, fail)
        error = OSError

    with pytest.raises(error):
        write_atomically(str(path), data)

    assert json.loads(path.read_text()) == {"key": "previous"}
    assert [p.name for p in tmp_path.iterdir()] == [path.name]

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

import pytest

from nvflare.app_opt.job_launcher.slurm.config import (
    SlurmLauncherError,
    _require_int,
    _require_positive_number,
    _validate_mount_source,
    normalize_multi_node_port_range,
    normalize_slurm_executables,
    normalize_slurm_image,
)


def test_mount_source_rejects_launcher_owned_path(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "data"
    source.mkdir()

    with pytest.raises(SlurmLauncherError, match="outside workspace_path"):
        _validate_mount_source(str(source), str(workspace), "mount source")


def test_executables_reject_unknown_key():
    with pytest.raises(SlurmLauncherError, match="unknown key"):
        normalize_slurm_executables({"unknown": "/usr/bin/true"})


def test_integer_rejects_boolean():
    with pytest.raises(SlurmLauncherError, match="must be an integer"):
        _require_int(True, "nodes")


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), float("-inf")])
def test_positive_number_rejects_non_finite_values_and_boolean(value):
    with pytest.raises(SlurmLauncherError, match="finite positive number"):
        _require_positive_number(value, "timeout")


def test_image_requires_existing_file_when_requested(tmp_path):
    with pytest.raises(SlurmLauncherError, match="not an existing regular file"):
        normalize_slurm_image(str(tmp_path / "missing.sif"), "apptainer", require_file=True)


def test_multi_node_port_range_is_normalized():
    assert normalize_multi_node_port_range("29400-29499", internal_port=8102) == (29400, 29499)
    assert normalize_multi_node_port_range((29400, 29499), internal_port=8102) == (29400, 29499)


@pytest.mark.parametrize("value", ["invalid", "900-1000", "30000-29999", "29400-70000"])
def test_multi_node_port_range_rejects_invalid_values(value):
    with pytest.raises(SlurmLauncherError, match="multi_node_port_range"):
        normalize_multi_node_port_range(value, internal_port=8102)


def test_multi_node_port_range_must_not_overlap_internal_port():
    with pytest.raises(SlurmLauncherError, match="internal_port"):
        normalize_multi_node_port_range("8000-8200", internal_port=8102)

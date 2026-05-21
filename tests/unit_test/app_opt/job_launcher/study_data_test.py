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

import logging

import pytest
import yaml

from nvflare.app_opt.job_launcher.study_data import (
    load_study_data_file,
    resolve_study_dataset_mounts,
    should_mount_study_data,
)


def _write_yaml(path, data):
    path.write_text(yaml.safe_dump(data))


def test_load_missing_file_returns_empty_mapping(tmp_path):
    assert load_study_data_file(str(tmp_path / "missing.yaml")) == {}


def test_load_missing_file_logs_warning_when_logger_supplied(tmp_path, caplog):
    logger = logging.getLogger("study-data-test")
    with caplog.at_level(logging.WARNING, logger="study-data-test"):
        assert load_study_data_file(str(tmp_path / "missing.yaml"), logger=logger) == {}

    assert "was not found" in caplog.text
    assert "no study data mounts will be configured" in caplog.text


def test_load_empty_file_logs_warning_when_logger_supplied(tmp_path, caplog):
    path = tmp_path / "study_data.yaml"
    path.write_text("")
    logger = logging.getLogger("study-data-test")

    with caplog.at_level(logging.WARNING, logger="study-data-test"):
        assert load_study_data_file(str(path), logger=logger) == {}

    assert "has no study entries" in caplog.text


def test_load_unreadable_file_raises_value_error(monkeypatch, tmp_path):
    path = tmp_path / "study_data.yaml"
    path.write_text("{}")

    def _raise_permission_error(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("builtins.open", _raise_permission_error)

    with pytest.raises(ValueError, match="Could not read study data file"):
        load_study_data_file(str(path))


def test_load_malformed_yaml_raises_value_error(tmp_path):
    path = tmp_path / "study_data.yaml"
    path.write_text("study-a: [")

    with pytest.raises(ValueError, match="Could not parse study data file"):
        load_study_data_file(str(path))


def test_load_accepts_nested_study_dataset_mapping(tmp_path):
    path = tmp_path / "study_data.yaml"
    data = {"study-a": {"training": {"source": "/data/train", "mode": "ro"}}}
    _write_yaml(path, data)

    assert load_study_data_file(str(path)) == data


def test_load_rejects_non_dict_root(tmp_path):
    path = tmp_path / "study_data.yaml"
    path.write_text("[]")

    with pytest.raises(ValueError, match="dictionary"):
        load_study_data_file(str(path))


def test_load_rejects_legacy_flat_mapping(tmp_path):
    path = tmp_path / "study_data.yaml"
    _write_yaml(path, {"study-a": "/data/train"})

    with pytest.raises(ValueError, match="study -> dataset"):
        load_study_data_file(str(path))


def test_load_rejects_dataset_entry_without_source(tmp_path):
    path = tmp_path / "study_data.yaml"
    _write_yaml(path, {"study-a": {"training": {"mode": "ro"}}})

    with pytest.raises(ValueError, match="source"):
        load_study_data_file(str(path))


def test_load_rejects_dataset_entry_with_invalid_mode(tmp_path):
    path = tmp_path / "study_data.yaml"
    _write_yaml(path, {"study-a": {"training": {"source": "/data/train", "mode": "read-only"}}})

    with pytest.raises(ValueError, match="mode"):
        load_study_data_file(str(path))


def test_resolve_study_dataset_mounts_returns_mount_metadata():
    study_data = {
        "study-a": {
            "training": {"source": "/data/train", "mode": "ro"},
            "output": {"source": "/data/out", "mode": "rw"},
        }
    }

    mounts = resolve_study_dataset_mounts(study_data, "study-a", "study_data.yaml")

    assert [m.mount_path for m in mounts] == ["/data/study-a/training", "/data/study-a/output"]
    assert [m.source for m in mounts] == ["/data/train", "/data/out"]
    assert [m.read_only for m in mounts] == [True, False]


def test_resolve_returns_empty_when_study_mapping_is_missing():
    assert resolve_study_dataset_mounts({}, "study-a", "study_data.yaml") == []


def test_resolve_logs_warning_when_study_mapping_is_missing(caplog):
    logger = logging.getLogger("study-data-test")
    study_data = {"other-study": {"training": {"source": "/data/train", "mode": "ro"}}}

    with caplog.at_level(logging.WARNING, logger="study-data-test"):
        assert resolve_study_dataset_mounts(study_data, "study-a", "study_data.yaml", logger=logger) == []

    assert "has no entry for study 'study-a'" in caplog.text


def test_resolve_returns_empty_for_unmapped_invalid_study_name():
    assert resolve_study_dataset_mounts({}, "Study_A", "study_data.yaml") == []


def test_resolve_returns_empty_when_study_mapping_is_empty():
    assert resolve_study_dataset_mounts({"study-a": {}}, "study-a", "study_data.yaml") == []


def test_resolve_logs_warning_when_study_mapping_is_empty(caplog):
    logger = logging.getLogger("study-data-test")

    with caplog.at_level(logging.WARNING, logger="study-data-test"):
        assert resolve_study_dataset_mounts({"study-a": {}}, "study-a", "study_data.yaml", logger=logger) == []

    assert "entry for study 'study-a' has no datasets" in caplog.text


def test_resolve_default_study_mapping_when_present():
    study_data = {"default": {"training": {"source": "/data/train", "mode": "ro"}}}

    mounts = resolve_study_dataset_mounts(study_data, "default", "study_data.yaml")

    assert len(mounts) == 1
    assert mounts[0].mount_path == "/data/default/training"
    assert mounts[0].source == "/data/train"


@pytest.mark.parametrize("study", [None, ""])
def test_default_or_missing_study_does_not_mount_data(study):
    assert should_mount_study_data(study) is False


@pytest.mark.parametrize("study", ["default", "study-a"])
def test_named_study_mounts_data_when_mapping_exists(study):
    assert should_mount_study_data(study) is True

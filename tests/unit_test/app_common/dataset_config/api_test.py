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

from unittest.mock import MagicMock

import pytest
import yaml

from nvflare.apis.dataset_config_spec import (
    DATASET_CONFIG_ENV,
    DatasetConfigError,
    UnknownDatasetError,
    get_study_dataset_config,
)
from nvflare.apis.fl_constant import FLContextKey
from nvflare.app_common.dataset_config.api import load_study_config, parse_study_config

CONFIG = {
    "format_version": 1,
    "studies": {
        "oncology": {
            "datasets": {
                "omop": {
                    "type": "parquet",
                    "uri": "file:///data/oncology/omop",
                    "usage": {"catalog": {"share": True, "exclude_tables": ["t"]}},
                },
                "ehr": {
                    "type": "sql",
                    "uri": "postgresql+psycopg://db.example:5432/ehr",
                    "options": {"schema": "public", "tables": ["visits"]},
                    "secrets": {"password": "EHR_DB_PASSWORD"},
                },
            }
        },
        "other": {"datasets": {"hidden": {"type": "csv", "uri": "file:///data/other/hidden.csv"}}},
    },
    "usage": {"catalog": {"limits": {"max_records": 50}}},
}


def parse(data, study="oncology"):
    return parse_study_config(data, study, "datasets.yaml")


def with_dataset(**entry):
    return {"format_version": 1, "studies": {"oncology": {"datasets": {"d": entry}}}}


class TestSnapshot:
    def test_study_view(self):
        config = parse(CONFIG)
        assert config.study == "oncology"
        assert config.list_datasets() == ["omop", "ehr"]
        omop = config.get_dataset("omop")
        assert (omop.type, omop.uri, omop.path) == ("parquet", "file:///data/oncology/omop", "/data/oncology/omop")
        assert omop.usage_rules("catalog")["share"] is True
        assert omop.usage_rules("cohort") is None
        ehr = config.get_dataset("ehr")
        assert ehr.path is None and ehr.secret_keys == ("password",)
        assert ehr.options["tables"] == ("visits",)
        assert config.site_usage("catalog")["limits"]["max_records"] == 50
        assert config.site_usage("cohort") is None

    def test_other_studies_are_not_visible(self):
        config = parse(CONFIG)
        assert "hidden" not in config.list_datasets()
        with pytest.raises(UnknownDatasetError):
            config.get_dataset("hidden")

    def test_absent_study_has_no_datasets(self):
        assert parse(CONFIG, study="default").list_datasets() == []

    def test_snapshot_is_read_only(self):
        omop = parse(CONFIG).get_dataset("omop")
        with pytest.raises(TypeError):
            omop.usage["catalog"]["share"] = False
        with pytest.raises(TypeError):
            omop.options["x"] = 1

    def test_secret_is_read_at_call_time(self, monkeypatch):
        config = parse(CONFIG)
        monkeypatch.delenv("EHR_DB_PASSWORD", raising=False)
        with pytest.raises(DatasetConfigError) as e:
            config.get_secret("ehr", "password")
        monkeypatch.setenv("EHR_DB_PASSWORD", "s3cret")
        assert config.get_secret("ehr", "password") == "s3cret"
        with pytest.raises(DatasetConfigError):
            config.get_secret("ehr", "token")
        assert "s3cret" not in str(e.value)

    def test_percent_encoded_path_is_decoded(self):
        config = parse(with_dataset(type="csv", uri="file:///data/a%20b.csv"))
        assert config.get_dataset("d").path == "/data/a b.csv"


class TestValidation:
    @pytest.mark.parametrize(
        "data",
        [
            [],
            {"format_version": 2},
            {"format_version": True},
            {"format_version": 1, "extra": 1},
            {"format_version": 1, "studies": {"Bad Name": {}}},
            {"format_version": 1, "studies": {"oncology": {"description": "x"}}},
            with_dataset(uri="file:///x"),
            with_dataset(type="parquet"),
            with_dataset(type="parquet", uri="data/omop"),
            with_dataset(type="parquet", uri="file://data/omop"),
            with_dataset(type="parquet", uri="file:///data/omop?x=1"),
            with_dataset(type="parquet", uri="file:relative/path"),
            with_dataset(type="sql", uri="postgresql://user:pw@host/db"),
            with_dataset(type="sql", uri="postgresql://host/db", secrets={"password": "not a variable!"}),
            with_dataset(type="parquet", uri="file:///x", extra=1),
            with_dataset(type="parquet", uri="file:///x", options=[]),
            {"format_version": 1, "studies": {"oncology": {"datasets": {"-bad": {"type": "x", "uri": "file:///x"}}}}},
        ],
    )
    def test_invalid(self, data):
        with pytest.raises(DatasetConfigError):
            parse(data)

    def test_a_broken_entry_in_another_study_invalidates_the_file(self):
        data = {**CONFIG, "studies": {**CONFIG["studies"], "other": {"datasets": {"x": {"type": "csv"}}}}}
        with pytest.raises(DatasetConfigError):
            parse(data)

    def test_username_without_password_is_allowed(self):
        config = parse(with_dataset(type="sql", uri="postgresql://reader@host/db"))
        assert config.get_dataset("d").path is None


class TestLoading:
    def test_missing_file(self, tmp_path):
        assert load_study_config(str(tmp_path / "none.yaml"), "oncology", missing_ok=True).list_datasets() == []
        with pytest.raises(DatasetConfigError):
            load_study_config(str(tmp_path / "none.yaml"), "oncology", missing_ok=False)

    def test_revision_changes_with_content(self, tmp_path):
        path = tmp_path / "datasets.yaml"
        path.write_text(yaml.safe_dump(CONFIG, sort_keys=False))
        first = load_study_config(str(path), "oncology", missing_ok=False).revision
        path.write_text(yaml.safe_dump({**CONFIG, "usage": {}}))
        assert load_study_config(str(path), "oncology", missing_ok=False).revision != first

    def test_yaml_anchors_are_resolved(self, tmp_path):
        path = tmp_path / "datasets.yaml"
        path.write_text(
            "format_version: 1\n"
            "studies:\n"
            "  oncology:\n"
            "    datasets:\n"
            "      a:\n"
            "        <<: &db {type: sql, uri: 'postgresql://h/db'}\n"
            "        options: {tables: [x]}\n"
            "      b:\n"
            "        <<: *db\n"
        )
        config = load_study_config(str(path), "oncology", missing_ok=False)
        assert config.get_dataset("b").type == "sql" and config.get_dataset("b").uri == "postgresql://h/db"

    def test_invalid_yaml(self, tmp_path):
        path = tmp_path / "datasets.yaml"
        path.write_text("format_version: [1\n")
        with pytest.raises(DatasetConfigError):
            load_study_config(str(path), "oncology", missing_ok=False)


def make_fl_ctx(workspace_root, study="oncology"):
    fl_ctx = MagicMock()
    fl_ctx.get_prop.side_effect = lambda key, default=None: {"study": study} if key == FLContextKey.JOB_META else None
    fl_ctx.get_engine.return_value.get_workspace.return_value.get_root_dir.return_value = str(workspace_root)
    return fl_ctx


class TestGetStudyDatasetConfig:
    def test_reads_the_site_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv(DATASET_CONFIG_ENV, raising=False)
        (tmp_path / "local").mkdir()
        (tmp_path / "local" / "datasets.yaml").write_text(yaml.safe_dump(CONFIG, sort_keys=False))
        assert get_study_dataset_config(make_fl_ctx(tmp_path)).list_datasets() == ["omop", "ehr"]

    def test_missing_site_file_is_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv(DATASET_CONFIG_ENV, raising=False)
        assert get_study_dataset_config(make_fl_ctx(tmp_path)).list_datasets() == []

    def test_delivered_file_is_used_and_must_exist(self, tmp_path, monkeypatch):
        delivered = tmp_path / "study.yaml"
        delivered.write_text(yaml.safe_dump(CONFIG, sort_keys=False))
        monkeypatch.setenv(DATASET_CONFIG_ENV, str(delivered))
        assert get_study_dataset_config(make_fl_ctx(tmp_path / "ws")).list_datasets() == ["omop", "ehr"]
        monkeypatch.setenv(DATASET_CONFIG_ENV, str(tmp_path / "missing.yaml"))
        with pytest.raises(DatasetConfigError):
            get_study_dataset_config(make_fl_ctx(tmp_path / "ws"))

    def test_relative_delivered_path_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv(DATASET_CONFIG_ENV, "study.yaml")
        with pytest.raises(DatasetConfigError):
            get_study_dataset_config(make_fl_ctx(tmp_path))

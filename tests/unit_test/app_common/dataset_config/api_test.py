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

import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from nvflare.apis.dataset_config_spec import DatasetConfigError, UnknownDatasetError, get_study_dataset_config
from nvflare.apis.fl_constant import FLContextKey
from nvflare.app_common.dataset_config import api as api_module
from nvflare.app_common.dataset_config.api import extract_study_config, load_study_config, parse_study_config

CONFIG = {
    "format_version": 1,
    "studies": {
        "oncology": {
            "datasets": {
                "omop": {
                    "type": "parquet",
                    "uri": "file:///data/oncology/omop",
                    "usage": {"training": {"shuffle": True}},
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
        assert omop.usage_rules("training")["shuffle"] is True
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
            omop.usage["training"]["shuffle"] = False
        with pytest.raises(TypeError):
            omop.options["x"] = 1

    def test_yaml_sets_are_read_only(self):
        data = yaml.safe_load(
            "format_version: 1\n"
            "studies: {oncology: {datasets: {d: {type: sql, uri: 'postgresql://h/db', options: {tables: !!set {visits: null}}}}}}\n"
        )
        tables = parse(data).get_dataset("d").options["tables"]
        assert tables == {"visits"}
        with pytest.raises(AttributeError):
            tables.clear()

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
            {"format_version": 1, "extra": 1, 7: 0},
            {"format_version": 1, "studies": {"oncology": {"datasets": {}, "extra": 0, 7: 0}}},
            {
                "format_version": 1,
                "studies": {"oncology": {"datasets": {"d": {"type": "x", "uri": "file:///x", "extra": 1, 7: 0}}}},
            },
            {"format_version": 1, "studies": {"Bad Name": {}}},
            {"format_version": 1, "studies": {"study\n": {}}},
            {"format_version": 1, "studies": {"oncology": {"datasets": {"d\n": {"type": "x", "uri": "file:///x"}}}}},
            with_dataset(type="sql", uri="postgresql://host/db", secrets={"password": "DB_PASSWORD\n"}),
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

    @pytest.mark.parametrize("rules", [False, True, [], None, "settings"])
    def test_component_settings_must_be_mappings_at_both_scopes(self, rules):
        for doc in (
            with_dataset(type="csv", uri="file:///x", usage={"catalog": rules}),
            {**CONFIG, "usage": {"catalog": rules}},
        ):
            with pytest.raises(DatasetConfigError, match="settings mappings"):
                parse(doc)

    def test_a_broken_entry_in_another_study_invalidates_the_file(self):
        data = {**CONFIG, "studies": {**CONFIG["studies"], "other": {"datasets": {"x": {"type": "csv"}}}}}
        with pytest.raises(DatasetConfigError):
            parse(data)

    def test_malformed_uri_is_a_config_error_that_does_not_echo_it(self):
        with pytest.raises(DatasetConfigError, match="is not a valid URI") as error:
            parse(with_dataset(type="sql", uri="postgresql://reader:s3cret@[::1/db"))
        assert "s3cret" not in str(error.value)

    @pytest.mark.parametrize(
        "uri",
        [
            "https://acct.blob.core.windows.net/c?sv=2024&sig=SECRET",  # Azure SAS
            "s3://bucket/key?X-Amz-Signature=SECRET",  # presigned
            "https://storage.googleapis.com/b/o?X-Goog-Signature=SECRET",
            "jdbc:postgresql://db:5432/x?user=u&password=SECRET",
            "https://api.example/v1?api_key=SECRET",
            "https://api.example/v1?Token=SECRET",
            "mssql+pyodbc:///?odbc_connect=DRIVER%3D%7BODBC+Driver+18%7D%3BPwD%3DSECRET",
            "mssql+pyodbc:///?ODBC_CONNECT=DSN%3Dapproved%3B+Password+%3D%7BSECRET%3Bvalue%7D",
            "mssql+pyodbc:///?odbc_connect=DSN%3Dapproved%253Bpwd%253DSECRET",
        ],
    )
    def test_credentials_in_the_query_are_rejected_without_echoing_them(self, uri):
        with pytest.raises(DatasetConfigError, match="carries a credential") as error:
            parse(with_dataset(type="sql", uri=uri))
        assert "SECRET" not in str(error.value)

    def test_odbc_settings_allow_braced_attributes_and_integrated_authentication(self):
        uri = "mssql+pyodbc:///?odbc_connect=DRIVER%3D%7Bdemo%3BPWD%3Dliteral%7D%3BTrusted_Connection%3Dyes"
        assert parse(with_dataset(type="sql", uri=uri)).get_dataset("d").uri == uri

    def test_invalid_odbc_attribute_is_rejected_without_echoing_the_string(self):
        with pytest.raises(DatasetConfigError, match="invalid odbc_connect") as error:
            parse(with_dataset(type="sql", uri="mssql+pyodbc:///?odbc_connect=DSN%3Dgood%3BSECRET"))
        assert "SECRET" not in str(error.value)

    def test_ordinary_query_parameters_are_allowed(self):
        config = parse(with_dataset(type="sql", uri="postgresql://db/x?sslmode=require&application_name=flare"))
        assert config.get_dataset("d").path is None

    @pytest.mark.parametrize("separator", ["/", "?", "#"])
    @pytest.mark.parametrize("prefix", ["pw", "123"])
    def test_unescaped_password_delimiter_is_rejected_without_echoing_it(self, separator, prefix):
        with pytest.raises(DatasetConfigError) as error:
            parse(with_dataset(type="sql", uri=f"postgresql://reader:{prefix}{separator}private@db/study"))
        assert "private" not in str(error.value)

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


def make_fl_ctx(workspace_root, study="oncology", job_id="job-1"):
    fl_ctx = MagicMock()
    fl_ctx.get_prop.side_effect = lambda key, default=None: {"study": study} if key == FLContextKey.JOB_META else None
    fl_ctx.get_job_id.return_value = job_id
    fl_ctx.get_engine.return_value.get_workspace.return_value.get_root_dir.return_value = str(workspace_root)
    return fl_ctx


def write_site_file(workspace_root, data):
    (workspace_root / "local").mkdir(parents=True, exist_ok=True)
    path = workspace_root / "local" / "datasets.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


class TestGetStudyDatasetConfig:
    def test_reads_local_datasets_yaml(self, tmp_path):
        write_site_file(tmp_path, CONFIG)
        assert get_study_dataset_config(make_fl_ctx(tmp_path)).list_datasets() == ["omop", "ehr"]

    def test_missing_file_is_empty(self, tmp_path):
        assert get_study_dataset_config(make_fl_ctx(tmp_path)).list_datasets() == []

    def test_loaded_once_per_job(self, tmp_path):
        path = write_site_file(tmp_path, CONFIG)
        first = get_study_dataset_config(make_fl_ctx(tmp_path, job_id="job-a"))
        path.write_text(yaml.safe_dump({"format_version": 1}))  # edited while job-a runs
        assert get_study_dataset_config(make_fl_ctx(tmp_path, job_id="job-a")) is first
        assert get_study_dataset_config(make_fl_ctx(tmp_path, job_id="job-b")).list_datasets() == []

    def test_a_failed_load_is_retried(self, tmp_path):
        path = write_site_file(tmp_path, {"format_version": 2})
        with pytest.raises(DatasetConfigError):
            get_study_dataset_config(make_fl_ctx(tmp_path, job_id="job-c"))
        path.write_text(yaml.safe_dump(CONFIG, sort_keys=False))
        assert get_study_dataset_config(make_fl_ctx(tmp_path, job_id="job-c")).list_datasets() == ["omop", "ehr"]

    def test_the_removed_environment_variable_is_ignored(self, tmp_path, monkeypatch):
        other = tmp_path / "elsewhere.yaml"
        other.write_text(yaml.safe_dump({"format_version": 1, "studies": {"oncology": {"datasets": {}}}}))
        monkeypatch.setenv("NVFLARE_DATASET_CONFIG", str(other))
        write_site_file(tmp_path, CONFIG)
        assert get_study_dataset_config(make_fl_ctx(tmp_path, job_id="job-env")).list_datasets() == ["omop", "ehr"]


class TestKubernetesStudyPart:
    def test_holds_only_the_jobs_study_and_site_usage(self, tmp_path):
        other = {
            **CONFIG,
            "studies": {**CONFIG["studies"], "cardiology": {"datasets": {"echo": {"type": "csv", "uri": "file:///e"}}}},
        }
        part = yaml.safe_load(extract_study_config(str(write_site_file(tmp_path, other)), "oncology"))
        assert list(part["studies"]) == ["oncology"]
        assert "usage" in CONFIG and part["usage"] == CONFIG["usage"]

    def test_a_job_sees_the_same_configuration_from_the_part_and_the_site_file(self, tmp_path):
        site = write_site_file(tmp_path / "site", CONFIG)
        job = tmp_path / "job"
        (job / "local").mkdir(parents=True)
        (job / "local" / "datasets.yaml").write_bytes(extract_study_config(str(site), "oncology"))
        from_site = load_study_config(str(site), "oncology", missing_ok=False)
        from_part = load_study_config(str(job / "local" / "datasets.yaml"), "oncology", missing_ok=False)
        assert from_part.list_datasets() == from_site.list_datasets() and from_part.revision == from_site.revision

    def test_another_study_gets_an_empty_part(self, tmp_path):
        part = yaml.safe_load(extract_study_config(str(write_site_file(tmp_path, CONFIG)), "cardiology"))
        assert part["studies"] == {}

    def test_no_site_file_sends_none(self, tmp_path):
        assert extract_study_config(str(tmp_path / "datasets.yaml"), "oncology") is None

    def test_mappings_with_non_string_keys_load_and_extract(self, tmp_path):
        data = with_dataset(type="csv", uri="file:///x", options={"columns": {1: "first", "other": "rest"}})
        site = write_site_file(tmp_path / "site", data)
        config = load_study_config(str(site), "oncology", missing_ok=False)
        assert config.revision == load_study_config(str(site), "oncology", missing_ok=False).revision
        part = extract_study_config(str(site), "oncology")
        assert yaml.safe_load(part)["studies"]["oncology"]["datasets"]["d"]["options"]["columns"][1] == "first"

    @pytest.mark.parametrize(
        "first, second",
        [
            ({"columns": {}}, {"columns": ["map", []]}),  # a mapping and a list that looks like its encoding
            ({"columns": {1: "a"}}, {"columns": {"1": "a"}}),
            ({"since": datetime.date(2026, 1, 1)}, {"since": "datetime.date(2026, 1, 1)"}),
            ({"tables": ["a", "b"]}, {"tables": ["b", "a"]}),
        ],
    )
    def test_different_settings_have_different_revisions(self, first, second):
        revisions = {
            parse(with_dataset(type="csv", uri="file:///x", options=options)).revision for options in (first, second)
        }
        assert len(revisions) == 2

    def test_an_invalid_other_study_is_not_revealed_to_the_job(self, tmp_path, caplog):
        bad = {"format_version": 1, "studies": {**CONFIG["studies"], "secret-study": {"datasets": {"hidden-set": {}}}}}
        site = write_site_file(tmp_path / "site", bad)
        with caplog.at_level("WARNING"):
            part = extract_study_config(str(site), "oncology")
        assert b"secret-study" not in part and b"hidden-set" not in part
        assert "secret-study" in caplog.text  # the reason stays at the site
        job_file = tmp_path / "datasets.yaml"
        job_file.write_bytes(part)
        with pytest.raises(DatasetConfigError) as error:
            load_study_config(str(job_file), "oncology", missing_ok=False)
        assert "secret-study" not in str(error.value) and "site's log" in str(error.value)

    def _rejected(self, tmp_path, part):
        job_file = tmp_path / "job-datasets.yaml"
        job_file.write_bytes(part)
        with pytest.raises(DatasetConfigError, match="site's log"):
            load_study_config(str(job_file), "oncology", missing_ok=False)

    def test_an_unreadable_site_file_sends_a_part_the_api_rejects(self, tmp_path, monkeypatch, caplog):
        site = write_site_file(tmp_path / "site", CONFIG)
        real_read = Path.read_bytes
        monkeypatch.setattr(
            Path,
            "read_bytes",
            lambda self: (_ for _ in ()).throw(PermissionError()) if self == site else real_read(self),
        )
        with caplog.at_level("WARNING"):
            part = extract_study_config(str(site), "oncology")
        assert "PermissionError" in caplog.text
        monkeypatch.setattr(Path, "read_bytes", real_read)
        self._rejected(tmp_path, part)

    def test_an_unexpected_error_never_escapes_and_sends_a_part_the_api_rejects(self, tmp_path, monkeypatch):
        site = write_site_file(tmp_path / "site", CONFIG)

        def broken(*args):
            raise TypeError("unexpected")

        monkeypatch.setattr(api_module, "_study_part", broken)
        part = extract_study_config(str(site), "oncology")  # must not raise: it runs in the workspace download
        monkeypatch.undo()
        self._rejected(tmp_path, part)

    def test_an_invalid_site_file_sends_a_part_the_api_rejects(self, tmp_path):
        site = write_site_file(
            tmp_path / "site", {"format_version": 1, "studies": {"oncology": {"datasets": {"d": {"type": "csv"}}}}}
        )
        job_file = tmp_path / "datasets.yaml"
        job_file.write_bytes(extract_study_config(str(site), "oncology"))
        with pytest.raises(DatasetConfigError, match="site's dataset configuration is invalid"):
            load_study_config(str(job_file), "oncology", missing_ok=False)

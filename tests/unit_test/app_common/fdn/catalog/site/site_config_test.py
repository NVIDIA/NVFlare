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

from nvflare.app_common.fdn.catalog.common.schema import ReadLimits
from nvflare.app_common.fdn.catalog.site.site_config import (
    DEFAULT_CONNECTOR_CLASS,
    PARQUET_CONNECTOR,
    ConnectorSpec,
    DatasetBinding,
    SiteConfigError,
    load_site_config,
)


def load(tmp_path, config):
    path = tmp_path / "catalog_site.json"
    path.write_text(json.dumps(config))
    return load_site_config(path)


class TestDefaults:
    def test_the_smallest_file_means_mounted_parquet_data_with_default_limits(self, tmp_path):
        site = load(tmp_path, {"format_version": 1})
        assert site.connectors[PARQUET_CONNECTOR] == ConnectorSpec(DEFAULT_CONNECTOR_CLASS)
        assert site.connectors["sqlite"].path == "nvflare.app_opt.fdn.catalog.sqlite_connector.SQLiteConnector"
        assert site.default_connector == PARQUET_CONNECTOR and site.bindings("any") == {}
        assert site.limits == ReadLimits(60, 1000, 10000) and site.installed_operations == frozenset()


class TestConnectorsAndBindings:
    CONFIG = {
        "format_version": 1,
        "connectors": {
            "claims-db": {"class": "a.Sql", "args": {"host": "h"}},
            "lake": {"class": "a.Lake"},
        },
        "studies": {
            "oncology": {
                "claims": {"connector": "claims-db", "options": {"schema": "public"}},
                "imaging-meta": {"connector": "lake"},
                "scans": {"connector": "parquet"},
            },
            "cardiology": {"echo": {"connector": "lake", "options": {"catalog": "c"}}},
        },
    }

    def test_the_built_in_parquet_connector_is_always_there_next_to_the_named_ones(self, tmp_path):
        site = load(tmp_path, self.CONFIG)
        assert set(site.connectors) == {"parquet", "sqlite", "claims-db", "lake"}
        assert site.connectors["claims-db"] == ConnectorSpec("a.Sql", {"host": "h"})

    def test_each_study_binds_its_own_datasets_to_connectors_of_different_source_types(self, tmp_path):
        site = load(tmp_path, self.CONFIG)
        assert site.bindings("oncology") == {
            "claims": DatasetBinding("claims-db", {"schema": "public"}),
            "imaging-meta": DatasetBinding("lake"),
            "scans": DatasetBinding("parquet"),
        }
        assert site.bindings("cardiology") == {"echo": DatasetBinding("lake", {"catalog": "c"})}
        assert site.bindings("other") == {}

    def test_a_site_can_name_its_own_default_connector_for_mounted_datasets(self, tmp_path):
        site = load(tmp_path, {**self.CONFIG, "default_connector": "lake"})
        assert site.default_connector == "lake"

    def test_a_file_can_replace_the_built_in_parquet_connector(self, tmp_path):
        site = load(tmp_path, {"format_version": 1, "connectors": {"parquet": {"class": "mine.Parquet"}}})
        assert site.connectors["parquet"].path == "mine.Parquet"

    @pytest.mark.parametrize(
        "change",
        [
            {"connectors": []},
            {"connectors": {"a": {}}},
            {"connectors": {"a": {"class": ""}}},
            {"connectors": {"a": {"class": "x.Y", "extra": 1}}},
            {"connectors": {"a": {"class": "x.Y", "args": []}}},
            {"connectors": {"bad name": {"class": "x.Y"}}},
            {"default_connector": "undefined"},
            {"studies": []},
            {"studies": {"s": []}},
            {"studies": {"": {}}},
            {"studies": {"s": {"Bad Name": {"connector": "parquet"}}}},
            {"studies": {"s": {"d": {}}}},
            {"studies": {"s": {"d": {"connector": "undefined"}}}},
            {"studies": {"s": {"d": {"connector": "parquet", "extra": 1}}}},
            {"studies": {"s": {"d": {"connector": "parquet", "options": []}}}},
        ],
    )
    def test_an_invalid_file_is_refused(self, tmp_path, change):
        with pytest.raises(SiteConfigError):
            load(tmp_path, {"format_version": 1, **change})

    def test_an_error_names_the_problem_and_never_the_values(self, tmp_path):
        with pytest.raises(SiteConfigError) as e:
            load(tmp_path, {"format_version": 1, "connectors": {"a": {"class": "x.Y", "args": "pw=hunter2"}}})
        assert "hunter2" not in str(e.value)

    @pytest.mark.parametrize("reference", [[], {}, None, True, 1, ""])
    @pytest.mark.parametrize("binding", [False, True])
    def test_invalid_connector_reference_types_are_configuration_errors(self, tmp_path, reference, binding):
        config = {"format_version": 1}
        if binding:
            config["studies"] = {"s": {"d": {"connector": reference}}}
        else:
            config["default_connector"] = reference
        with pytest.raises(SiteConfigError):
            load(tmp_path, config)

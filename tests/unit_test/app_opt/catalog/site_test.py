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

import os

import pytest
import yaml

from nvflare.app_opt.catalog.site import GENERATED_CONTRACT_KEYS, read_contracts, read_site_catalog
from nvflare.fuel.utils import fobs

from .catalog_test_utils import CONTRACT, FakeFLContext, write_site_config

PROVIDED = {"omop": {"contract": "omop.odcs.yaml"}}


def test_provided_contract_is_returned_as_written_and_can_be_sent(tmp_path):
    config_dir = write_site_config(tmp_path, PROVIDED)

    reply = read_contracts(config_dir, "site-1", "oncology")

    assert reply == {"contracts": {"omop": CONTRACT}, "unavailable": []}
    assert fobs.loads(fobs.dumps(reply)) == reply


def test_only_the_jobs_study_is_read(tmp_path):
    config_dir = write_site_config(tmp_path, PROVIDED, study="oncology")

    assert read_contracts(config_dir, "site-1", "cardiology") == {"contracts": {}, "unavailable": []}


def test_site_without_config_publishes_nothing(tmp_path):
    assert read_contracts(str(tmp_path), "site-1", "oncology") == {"contracts": {}, "unavailable": []}


def test_a_broken_dataset_does_not_cost_the_others(tmp_path):
    config_dir = write_site_config(tmp_path, {**PROVIDED, "claims": {"contract": "missing.yaml"}, "labs": {}})

    result = read_contracts(config_dir, "site-1", "oncology")

    assert list(result["contracts"]) == ["omop"]
    assert result["unavailable"] == ["claims", "labs"]


def test_generated_contract_keeps_only_the_schema(tmp_path):
    pytest.importorskip("datacontract")
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    source = str(tmp_path / "person.parquet")
    pq.write_table(pa.table({"person_id": [1], "year_of_birth": [1970]}), source)
    config_dir = write_site_config(tmp_path, {"omop": {"import": {"format": "parquet", "source": source}}})

    contract = yaml.safe_load(read_contracts(config_dir, "site-2", "oncology")["contracts"]["omop"])

    assert set(contract) <= set(GENERATED_CONTRACT_KEYS)
    assert (contract["kind"], contract["id"], contract["name"]) == ("DataContract", "site-2.omop", "omop")
    assert [p["name"] for p in contract["schema"][0]["properties"]] == ["person_id", "year_of_birth"]


def test_generation_gets_credentials_from_the_sites_environment(tmp_path, monkeypatch):
    data_contract = pytest.importorskip("datacontract.data_contract")
    seen = {}

    class Imported:
        def to_yaml(self):
            return "kind: DataContract\nservers: [{host: db.internal}]\nschema: []\n"

    def import_from_source(format, source=None, config=None, **kwargs):
        seen.update(format=format, source=source, config=config, kwargs=kwargs)
        return Imported()

    monkeypatch.setattr(data_contract.DataContract, "import_from_source", import_from_source)
    monkeypatch.setenv("CLAIMS_DB_PASSWORD", "s3cret")
    spec = {
        "import": {"format": "postgres", "source": "db.internal", "database": "claims"},
        "env": {"DATACONTRACT_POSTGRES_PASSWORD": "CLAIMS_DB_PASSWORD"},
    }
    config_dir = write_site_config(tmp_path, {"claims": spec})

    contract = yaml.safe_load(read_contracts(config_dir, "site-1", "oncology")["contracts"]["claims"])

    assert seen == {
        "format": "postgres",
        "source": "db.internal",
        "config": {"DATACONTRACT_POSTGRES_PASSWORD": "s3cret"},
        "kwargs": {"database": "claims"},
    }
    assert "servers" not in contract


def test_config_root_selects_the_site_folder_in_the_simulator(tmp_path):
    write_site_config(tmp_path / "site-2", PROVIDED, study="default")
    fl_ctx = FakeFLContext(tmp_path / "unused", site="site-2", study="default", simulate=True)

    assert list(read_site_catalog(fl_ctx, config_root=str(tmp_path))["contracts"]) == ["omop"]


def test_config_root_is_refused_outside_the_simulator(tmp_path):
    write_site_config(tmp_path / "site-1", PROVIDED)

    with pytest.raises(RuntimeError, match="simulator only"):
        read_site_catalog(FakeFLContext(os.devnull), config_root=str(tmp_path))

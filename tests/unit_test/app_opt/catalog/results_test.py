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
import os

import pytest

from nvflare.app_opt.catalog.results import download_catalog, write_results

from .catalog_test_utils import CONTRACT, FakeFLContext


def _reply(contracts, unavailable=()):
    return {"contracts": contracts, "unavailable": list(unavailable)}


def _write(tmp_path, replies):
    """Write the replies as job-1's result; return the catalog's file names and its parsed sites.json."""
    write_results(FakeFLContext(tmp_path), replies)
    catalog_dir = tmp_path / "job-1" / "catalog"
    return sorted(os.listdir(catalog_dir)), json.loads((catalog_dir / "sites.json").read_text())


def test_each_site_gets_a_status_and_a_file_per_contract(tmp_path):
    files, sites = _write(tmp_path, {"site-2": _reply({"omop": CONTRACT}, ["labs"]), "site-1": None})

    assert files == ["site-2.omop.odcs.yaml", "sites.json"]
    assert sites == {
        "site-1": {"status": "FAILED", "unavailable": []},
        "site-2": {"status": "OK", "unavailable": ["labs"]},
    }
    assert (tmp_path / "job-1" / "catalog" / "site-2.omop.odcs.yaml").read_text() == CONTRACT


@pytest.mark.parametrize("name", ["../escape", ""])
def test_a_dataset_name_that_is_not_a_safe_file_name_is_unavailable(tmp_path, name):
    files, sites = _write(tmp_path, {"site-1": _reply({name: CONTRACT})})

    assert sites["site-1"]["unavailable"] == [name]
    assert files == ["sites.json"]
    assert sorted(os.listdir(tmp_path)) == ["job-1"]


class _Run:
    job_id = "job-1"

    def __init__(self, result_dir):
        self.result_dir = result_dir

    def get_result(self, clean_up, outputs_only):
        return self.result_dir

    def get_status(self):
        return "FINISHED:COMPLETED"


def test_download_finds_the_catalog_folder_and_copies_it(tmp_path):
    catalog_dir = tmp_path / "download" / "workspace" / "catalog"
    os.makedirs(catalog_dir)
    (catalog_dir / "sites.json").write_text("{}")
    (catalog_dir / "site-1.omop.odcs.yaml").write_text(CONTRACT)
    run = _Run(str(tmp_path / "download"))
    output_dir = str(tmp_path / "mine")

    assert download_catalog(run) == str(catalog_dir)
    assert download_catalog(run, output_dir) == output_dir
    assert sorted(os.listdir(output_dir)) == ["site-1.omop.odcs.yaml", "sites.json"]


def test_download_reports_a_job_without_catalog():
    with pytest.raises(RuntimeError, match="job-1 has no catalog"):
        download_catalog(_Run(None))

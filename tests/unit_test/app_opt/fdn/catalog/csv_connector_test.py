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

import io
from pathlib import Path

import pytest

from nvflare.app_common.fdn.catalog.common.schema import ReadLimits, validate
from nvflare.app_common.fdn.catalog.site.connector import ConnectorError
from nvflare.app_opt.fdn.catalog.csv_connector import INFERENCE_BLOCK_BYTES, MAX_SAMPLE_BYTES, CSVConnector


def read(tmp_path, content, **limits):
    path = tmp_path / "measurements.csv"
    path.write_text(content)
    connector = CSVConnector({"measurements": path})
    return connector.read_catalog(
        "measurements", ReadLimits(**({"max_duration_s": 60, "max_calls": 1000, "max_records": 10000} | limits))
    )


def test_infers_schema_without_returning_source_rows(tmp_path):
    catalog = read(
        tmp_path,
        "id,value,date,flag,label,empty,code\n1,2.5,2024-01-01,true,private-marker,,001\n2,3,2024-01-02,false,other,,002\n",
    )
    validate(catalog, ReadLimits(60, 1000, 10000))
    columns = catalog.tables[0].columns
    assert [c.type for c in columns] == ["int64", "double", "date32[day]", "bool", "string", "null", "int64"]
    assert all(c.nullable is None for c in columns)
    assert catalog.provenance.source == "csv"
    assert "private-marker" not in str(catalog.to_wire())
    assert not catalog.tables[0].primary_key and not catalog.tables[0].relationships


@pytest.mark.parametrize("content", ["", "a,a\n1,2\n", "a,\n1,2\n", "a,b\n1\n", "a,b\n1,2,3\n"])
def test_invalid_header_or_sample_fails_with_safe_error(tmp_path, content):
    with pytest.raises(ConnectorError) as error:
        read(tmp_path, content)
    assert str(tmp_path) not in str(error.value)
    assert "CSV parse error" not in str(error.value)


@pytest.mark.parametrize("limits", [{"max_records": 2}, {"max_calls": 0}, {"max_duration_s": 0}])
def test_respects_site_budgets(tmp_path, limits):
    with pytest.raises(ConnectorError) as error:
        read(tmp_path, "value\n1\n2\n", **limits)
    assert error.value.budget


def test_inference_describes_initial_block_instead_of_later_rows(tmp_path):
    catalog = read(tmp_path, "value\n" + "1\n" * INFERENCE_BLOCK_BYTES + "word\n")
    assert catalog.tables[0].columns[0].type == "int64"


def test_conflicting_types_become_string(tmp_path):
    assert read(tmp_path, "value\n1\nword\n").tables[0].columns[0].type == "string"


def test_sample_bytes_are_bounded(tmp_path, monkeypatch):
    path = tmp_path / "measurements.csv"
    path.write_text("unused")

    class MeasuredSource(io.BytesIO):
        consumed = 0

        def read(self, size=-1):
            value = super().read(size)
            self.consumed += len(value)
            return value

    source = MeasuredSource(b"value\n" + b"1\n" * MAX_SAMPLE_BYTES)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: source)
    catalog = CSVConnector({"measurements": path}).read_catalog("measurements", ReadLimits(60, 1000, 10000))
    assert source.closed and source.consumed == MAX_SAMPLE_BYTES
    assert catalog.tables[0].columns[0].type == "int64"


def test_header_only_uses_arrow_null_type(tmp_path):
    assert read(tmp_path, "value\n").tables[0].columns[0].type == "null"


def test_arrow_handles_quoted_commas_multiline_fields_and_bom(tmp_path):
    catalog = read(tmp_path, '\ufeffid,note\n1,"a,b\nc"\n')
    assert [(c.name, c.type) for c in catalog.tables[0].columns] == [("id", "int64"), ("note", "string")]

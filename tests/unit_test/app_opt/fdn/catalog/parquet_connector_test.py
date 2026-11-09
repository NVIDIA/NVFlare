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
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nvflare.app_common.fdn.catalog.common.schema import ReadLimits, validate
from nvflare.app_common.fdn.catalog.site.connector import ConnectorError
from nvflare.app_opt.fdn.catalog.parquet_connector import ParquetConnector

LIMITS = ReadLimits(max_duration_s=30, max_calls=100, max_records=1000)


def write(path, **columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


@pytest.fixture
def root(tmp_path):
    write(tmp_path / "person.parquet", person_id=pa.array([1, 2], pa.int64()), name=pa.array(["a", "b"]))
    write(tmp_path / "visit" / "part-0.parquet", visit_id=pa.array([1], pa.int64()))
    write(tmp_path / "visit" / "part-1.parquet", visit_id=pa.array([2], pa.int64()))
    return tmp_path


def read(root, limits=LIMITS):
    return ParquetConnector({"ds": root}).read_catalog("ds", limits)


class TestSchema:
    def test_tables_columns_and_types_as_arrow_prints_them(self, root):
        catalog = read(root)
        assert [t.name for t in catalog.tables] == ["person", "visit"]
        person = catalog.tables[0]
        assert [(c.name, c.type) for c in person.columns] == [("person_id", "int64"), ("name", "string")]
        assert person.columns[0].nullable is True
        assert catalog.provenance.source == "parquet"
        assert catalog.display_name == "ds" and catalog.observed_at is None
        validate(catalog, LIMITS)

    def test_single_file_dataset(self, root):
        catalog = ParquetConnector({"ds": root / "person.parquet"}).read_catalog("ds", LIMITS)
        assert [t.name for t in catalog.tables] == ["person"]

    def test_single_file_container_alias_needs_no_extension(self, tmp_path):
        alias = tmp_path / "clinical"
        write(alias, person_id=[1])
        catalog = read(alias)
        assert [(t.name, [c.name for c in t.columns]) for t in catalog.tables] == [("clinical", ["person_id"])]

    def test_nested_hive_parts_form_one_table_and_every_footer_is_checked(self, tmp_path):
        first = tmp_path / "person" / "year=2025" / "part-0.parquet"
        second = tmp_path / "person" / "year=2026" / "month=01" / "part-1.parquet"
        write(first, person_id=[1])
        write(second, person_id=[2])
        catalog = read(tmp_path)
        assert [(t.name, [c.name for c in t.columns]) for t in catalog.tables] == [("person", ["person_id"])]
        with pytest.raises(ConnectorError) as stopped:
            read(tmp_path, ReadLimits(30, 1, 1000))
        assert stopped.value.budget
        write(second, other=[2])
        with pytest.raises(ConnectorError, match="inconsistent schemas"):
            read(tmp_path)
        second.write_bytes(b"corrupt")
        with pytest.raises(ConnectorError, match="unreadable"):
            read(tmp_path)

    def test_unknown_dataset_or_missing_mount(self, tmp_path):
        with pytest.raises(ConnectorError):
            ParquetConnector({}).read_catalog("ds", LIMITS)
        with pytest.raises(ConnectorError):
            ParquetConnector({"ds": tmp_path / "gone"}).read_catalog("ds", LIMITS)

    def test_corrupt_footer(self, root):
        (root / "bad.parquet").write_bytes(b"not parquet")
        with pytest.raises(ConnectorError) as e:
            read(root)
        assert not e.value.budget and str(root) not in str(e.value)

    def test_inconsistent_part_schemas(self, root):
        write(root / "visit" / "part-2.parquet", other=pa.array([1], pa.int64()))
        with pytest.raises(ConnectorError):
            read(root)


def test_an_unreadable_directory_is_a_safe_connector_error_that_names_no_path(tmp_path, monkeypatch):
    write(tmp_path / "events" / "part.parquet", value=[1])
    original = Path.iterdir

    def broken(self):
        if self.name == "events":
            raise PermissionError(f"denied: {self}")
        return original(self)

    monkeypatch.setattr(Path, "iterdir", broken)
    with pytest.raises(ConnectorError) as e:
        read(tmp_path)
    assert not e.value.budget and str(tmp_path) not in str(e.value)


def test_a_symlinked_directory_is_not_followed_so_it_cannot_add_a_table(tmp_path):
    outside = tmp_path / "outside"
    write(outside / "secret.parquet", value=[1])
    dataset = tmp_path / "dataset"
    write(dataset / "person.parquet", person_id=pa.array([1], pa.int64()))
    try:
        (dataset / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not available")
    assert [t.name for t in read(dataset).tables] == ["person"]


class TestBudgets:
    def test_the_walk_of_a_partitioned_table_stops_at_the_call_budget(self, tmp_path):
        for i in range(5):
            write(tmp_path / "events" / f"day={i}" / "part.parquet", value=[i])
        walked = []
        original = Path.iterdir

        def counting_iterdir(self):
            for p in original(self):
                if p.suffix == ".parquet":
                    walked.append(p)
                yield p

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Path, "iterdir", counting_iterdir)
            with pytest.raises(ConnectorError) as e:
                read(tmp_path, ReadLimits(30, 2, 1000))
        assert e.value.budget and len(walked) == 3  # stopped at the budget, not after listing the whole tree

    def test_call_budget(self, root):
        with pytest.raises(ConnectorError) as e:
            read(root, ReadLimits(30, 1, 1000))
        assert e.value.budget

    def test_unverified_parts_are_a_budget_failure_not_a_catalog(self, root):
        # three parts are listed (3 calls), then footers are read; the last footer is the sixth call
        with pytest.raises(ConnectorError) as e:
            read(root, ReadLimits(30, 5, 1000))
        assert e.value.budget

    def test_every_part_listed_and_verified_within_budget(self, root):
        # each of the three parts costs one call to list and one to read
        catalog = read(root, ReadLimits(30, 6, 1000))
        assert [t.name for t in catalog.tables] == ["person", "visit"]

    def test_record_budget(self, root):
        with pytest.raises(ConnectorError) as e:
            read(root, ReadLimits(30, 100, 3))
        assert e.value.budget


class TestSidecar:
    def sidecar(self, root, data):
        (root / "catalog.json").write_text(json.dumps(data))

    def test_declarations_are_carried(self, root):
        self.sidecar(
            root,
            {
                "display_name": "Clinical",
                "owner": "steward",
                "modalities": ["clinical_tabular"],
                "operations": ["cohort.count"],
                "semantic_profile": {"standard": "OMOP", "version": "5.4"},
                "provenance": {"declaration_ref": "decl-1"},
                "observed_at": "2026-01-01T00:00:00Z",
                "tables": {"person": {"primary_key": ["person_id"]}},
            },
        )
        catalog = read(root)
        assert catalog.display_name == "Clinical" and catalog.owner == "steward"
        assert catalog.semantic_profile.standard == "OMOP"
        assert catalog.provenance.declaration_ref == "decl-1" and catalog.provenance.source == "parquet"
        assert catalog.tables[0].primary_key == ("person_id",)
        assert catalog.observed_at == "2026-01-01T00:00:00Z"
        validate(catalog, LIMITS)

    @pytest.mark.parametrize(
        "data",
        [
            {"unknown": 1},
            {"provenance": {"source": "elsewhere"}},
            {"semantic_profile": {"bogus": "x"}},
            {"tables": {"person": {"extra": []}}},
            {"modalities": "not a list"},
            [],
        ],
    )
    def test_invalid_sidecar_is_a_connector_error(self, root, data):
        self.sidecar(root, data)
        with pytest.raises(ConnectorError):
            read(root)

    def test_oversize_and_unreadable_sidecar(self, root):
        (root / "catalog.json").write_text("x" * (1024 * 1024 + 1))
        with pytest.raises(ConnectorError):
            read(root)
        (root / "catalog.json").write_text("{not json")
        with pytest.raises(ConnectorError):
            read(root)

    def test_file_mtime_is_never_the_observation(self, root):
        assert read(root).observed_at is None


@pytest.mark.parametrize("nested", [False, True])
def test_walk_deadline_applies_without_any_parquet_matches(tmp_path, monkeypatch, nested):
    from nvflare.app_opt.fdn.catalog import parquet_connector as module
    from tests.timing_utils import ManualClock

    clock = ManualClock(now=0)
    monkeypatch.setattr(module, "time", clock)
    directory = tmp_path / "table" if nested else tmp_path
    directory.mkdir(exist_ok=True)
    for i in range(10):
        (directory / str(i)).mkdir()
    original = Path.iterdir
    visited = []

    def slow(directory):
        for path in original(directory):
            clock.advance(0.4)
            visited.append(path)
            yield path

    monkeypatch.setattr(Path, "iterdir", slow)
    with pytest.raises(ConnectorError) as error:
        read(tmp_path, ReadLimits(1, 100, 1000))
    assert error.value.budget and len(visited) == 3

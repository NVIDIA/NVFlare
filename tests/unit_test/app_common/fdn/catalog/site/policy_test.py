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

from nvflare.app_common.fdn.catalog.common.schema import ColumnInfo, Provenance, Relationship, TableInfo, validate
from nvflare.app_common.fdn.catalog.site.policy import (
    DatasetRule,
    PolicyError,
    apply_policy,
    load_policy,
    policy_file_path,
)
from tests.unit_test.app_common.fdn.catalog.helpers import LIMITS, make_catalog


def write_policy(tmp_path, data) -> str:
    path = tmp_path / "catalog_policy.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


class TestLoadPolicy:
    def test_valid_file(self, tmp_path):
        path = write_policy(
            tmp_path,
            {
                "format_version": 1,
                "datasets": {
                    "a": {
                        "share": True,
                        "exclude_tables": ["t"],
                        "exclude_columns": {"person": ["name"]},
                        "exclude_resources": ["Patient"],
                        "exclude_metadata": ["description"],
                        "operations": ["cohort.count"],
                    },
                    "b": {"share": False},
                },
            },
        )
        policy = load_policy(path)
        assert policy.datasets["a"].exclude_columns == {"person": frozenset({"name"})}
        assert policy.datasets["b"].share is False
        assert "unlisted" not in policy.datasets

    def test_missing_file(self, tmp_path):
        with pytest.raises(PolicyError):
            load_policy(tmp_path / "missing.json")

    @pytest.mark.parametrize(
        "data",
        [
            "not json",
            [],
            {"format_version": 2},
            {"format_version": True},
            {"format_version": 1.0},
            {"format_version": 1, "extra": {}},
            {"format_version": 1, "datasets": {"a": {}}},
            {"format_version": 1, "datasets": {"a": {"share": "yes"}}},
            {"format_version": 1, "datasets": {"a": {"share": True, "unknown": 1}}},
            {"format_version": 1, "datasets": {"a": {"share": True, "exclude_metadata": ["display_name"]}}},
            {"format_version": 1, "datasets": {"a": {"share": True, "exclude_tables": "t"}}},
            {"format_version": 1, "datasets": {"a": {"share": True, "exclude_columns": {"t": "c"}}}},
            {"format_version": 1, "derived": {"share": True}},
        ],
    )
    def test_invalid_file(self, tmp_path, data):
        with pytest.raises(PolicyError):
            load_policy(write_policy(tmp_path, data))

    def test_not_cached(self, tmp_path):
        path = write_policy(tmp_path, {"format_version": 1, "datasets": {"a": {"share": True}}})
        assert load_policy(path).datasets["a"].share is True
        write_policy(tmp_path, {"format_version": 1, "datasets": {"a": {"share": False}}})
        assert load_policy(path).datasets["a"].share is False


RULE = DatasetRule(share=True, operations=frozenset({"cohort.count", "cohort.sum"}))


class TestApplyPolicy:
    def test_input_is_not_mutated(self):
        catalog = make_catalog()
        apply_policy(DatasetRule(share=True, exclude_tables=frozenset({"person"})), catalog, {"cohort.count"})
        assert catalog == make_catalog()

    def test_excluded_table_removes_dependent_relationship(self):
        rule = DatasetRule(share=True, exclude_tables=frozenset({"person"}))
        view = apply_policy(rule, make_catalog(), set())
        assert [t.name for t in view.tables] == ["visit"]
        assert view.tables[0].relationships == ()
        validate(view, LIMITS)

    def test_excluded_column_empties_a_broken_primary_key(self):
        rule = DatasetRule(share=True, exclude_columns={"person": frozenset({"person_id"})})
        view = apply_policy(rule, make_catalog(), set())
        person = next(t for t in view.tables if t.name == "person")
        assert person.primary_key == () and [c.name for c in person.columns] == ["name"]
        # the relationship targeting the removed column is dropped, the view still validates
        assert next(t for t in view.tables if t.name == "visit").relationships == ()
        validate(view, LIMITS)

    def test_excluded_source_column_drops_its_relationship(self):
        rule = DatasetRule(share=True, exclude_columns={"visit": frozenset({"person_id"})})
        view = apply_policy(rule, make_catalog(), set())
        assert next(t for t in view.tables if t.name == "visit").relationships == ()

    def test_unrelated_broken_reference_is_kept_for_validation(self):
        broken = TableInfo("t", (ColumnInfo("a", "int"),), relationships=(Relationship(("a",), "missing", ("x",)),))
        view = apply_policy(DatasetRule(share=True), make_catalog(tables=(broken,)), set())
        assert view.tables[0].relationships  # policy removed nothing, validate will fail it

    def test_name_matching_nothing_is_ignored(self):
        rule = DatasetRule(share=True, exclude_tables=frozenset({"nope"}), exclude_columns={"nope": frozenset({"x"})})
        assert apply_policy(rule, make_catalog(), set()).tables == make_catalog().tables

    def test_excluded_resource(self):
        rule = DatasetRule(share=True, exclude_resources=frozenset({"Patient"}))
        assert apply_policy(rule, make_catalog(), set()).resources == ()

    def test_exclude_metadata(self):
        catalog = make_catalog(description="d", owner="o", source_version="v1")
        rule = DatasetRule(
            share=True,
            exclude_metadata=frozenset(
                {"description", "owner", "modalities", "semantic_profile", "source_version", "lineage", "provenance"}
            ),
        )
        view = apply_policy(rule, catalog, set())
        assert view.description is None and view.owner is None and view.source_version is None
        assert view.modalities == () and view.lineage == ()
        assert view.semantic_profile.standard is None
        assert view.provenance == Provenance(source="parquet")
        assert view.display_name == catalog.display_name

    def test_operations_are_the_intersection_sorted(self):
        catalog = make_catalog(operations=("cohort.sum", "cohort.count", "other.op"))
        view = apply_policy(RULE, catalog, {"cohort.count", "cohort.sum", "other.op"})
        assert view.operations == ("cohort.count", "cohort.sum")
        assert apply_policy(RULE, catalog, {"cohort.count"}).operations == ("cohort.count",)
        assert apply_policy(DatasetRule(share=True), catalog, {"cohort.count"}).operations == ()


@pytest.mark.parametrize("study", ["../other", "", "/tmp/other", "..", "x/y", "x\\y"])
def test_policy_path_rejects_unsafe_study_names(tmp_path, study):
    with pytest.raises(PolicyError):
        policy_file_path(tmp_path, study)


def test_studies_resolve_distinct_explicit_policy_files(tmp_path):
    assert policy_file_path(tmp_path, "default") == tmp_path / "default/catalog_policy.json"
    assert policy_file_path(tmp_path, "oncology") == tmp_path / "oncology/catalog_policy.json"

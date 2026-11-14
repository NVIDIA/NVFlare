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

from nvflare.app_common.fdn.catalog.common.schema import (
    DEFAULT_LIMITS,
    ColumnInfo,
    Provenance,
    ReadLimits,
    Relationship,
    TableInfo,
    validate,
)
from nvflare.app_common.fdn.catalog.site.rules import DatasetRule, RulesError, apply_policy, parse_rule, parse_settings
from tests.unit_test.app_common.fdn.catalog.helpers import LIMITS, make_catalog


class TestParseRule:
    def test_valid_rule(self):
        rule = parse_rule(
            {
                "share": True,
                "exclude_tables": ["t"],
                "exclude_columns": {"person": ["name"]},
                "exclude_resources": ["Patient"],
                "exclude_metadata": ["description"],
                "operations": ["cohort.count"],
                "connector": "duckdb",
            }
        )
        assert rule.exclude_columns == {"person": frozenset({"name"})}
        assert rule.connector == "duckdb"
        assert parse_rule({"share": False}).share is False

    @pytest.mark.parametrize(
        "value",
        [
            [],
            {},
            {"share": "yes"},
            {"share": True, "unknown": 1},
            {"share": True, "exclude_metadata": ["display_name"]},
            {"share": True, "exclude_tables": "t"},
            {"share": True, "exclude_columns": {"t": "c"}},
            {"share": True, "connector": "../x"},
        ],
    )
    def test_invalid_rule(self, value):
        with pytest.raises(RulesError):
            parse_rule(value)

    @pytest.mark.parametrize(
        "value",
        [
            {"share": True, "exclude_columns": {"secret_table": "secret_column"}},
            {"share": True, "secret_table": ["x"]},
            {"share": True, "exclude_metadata": ["secret_field"]},
        ],
    )
    def test_errors_never_repeat_names_the_site_typed(self, value):
        with pytest.raises(RulesError) as e:
            parse_rule(value)
        assert "secret" not in str(e.value)


class TestParseSettings:
    def test_none_is_all_defaults(self):
        settings = parse_settings(None)
        assert settings.limits == ReadLimits(**DEFAULT_LIMITS)
        assert settings.connectors == {} and settings.installed_operations == frozenset()

    def test_valid_settings(self):
        settings = parse_settings(
            {
                "limits": {"max_records": 50},
                "connectors": {"duckdb": {"class": "pkg.DuckDBConnector", "args": {"x": 1}}},
                "installed_operations": ["cohort.count"],
            }
        )
        assert settings.limits.max_records == 50 and settings.limits.max_calls == DEFAULT_LIMITS["max_calls"]
        assert settings.connectors["duckdb"].path == "pkg.DuckDBConnector"
        assert settings.connectors["duckdb"].args == {"x": 1}
        assert settings.installed_operations == frozenset({"cohort.count"})

    @pytest.mark.parametrize(
        "value",
        [
            [],
            {"unknown": 1},
            {"limits": {"max_records": 0}},
            {"limits": {"max_records": True}},
            {"limits": {"other": 1}},
            {"connectors": {"x": {"args": {}}}},
            {"connectors": {"x": {"class": "a.B", "extra": 1}}},
            {"connectors": {"x": {"class": "a.B", "args": []}}},
            {"installed_operations": "cohort.count"},
        ],
    )
    def test_invalid_settings(self, value):
        with pytest.raises(RulesError):
            parse_settings(value)


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

    def test_a_relationship_target_is_matched_without_regard_to_case(self):
        visit = TableInfo(
            "visit",
            (ColumnInfo("person_id", "int"),),
            relationships=(Relationship(("person_id",), "PERSON", ("person_id",)),),
        )
        catalog = make_catalog(tables=(make_catalog().tables[0], visit))
        view = apply_policy(DatasetRule(share=True, exclude_tables=frozenset({"person"})), catalog, set())
        assert next(t for t in view.tables if t.name == "visit").relationships == ()

    def test_unrelated_broken_reference_is_kept_for_validation(self):
        broken = TableInfo("t", (ColumnInfo("a", "int"),), relationships=(Relationship(("a",), "missing", ("x",)),))
        view = apply_policy(DatasetRule(share=True), make_catalog(tables=(broken,)), set())
        assert view.tables[0].relationships  # policy removed nothing, validate will fail it

    def test_name_matching_nothing_is_ignored(self):
        rule = DatasetRule(share=True, exclude_tables=frozenset({"nope"}), exclude_columns={"nope": frozenset({"x"})})
        assert apply_policy(rule, make_catalog(), set()).tables == make_catalog().tables

    def test_names_are_matched_without_regard_to_case(self):
        rule = DatasetRule(
            share=True, exclude_tables=frozenset({"VISIT"}), exclude_columns={"PERSON": frozenset({"NAME"})}
        )
        view = apply_policy(rule, make_catalog(), set())
        assert [t.name for t in view.tables] == ["person"]
        assert [c.name for c in view.tables[0].columns] == ["person_id"]

    def test_an_exclusion_that_matches_nothing_is_reported_and_excludes_nothing(self, caplog):
        rule = DatasetRule(
            share=True,
            exclude_tables=frozenset({"patients"}),
            exclude_columns={"person": frozenset({"ssn"}), "visits": frozenset({"x"})},
        )
        with caplog.at_level("WARNING", logger="nvflare.app_common.fdn.catalog.site.policy"):
            view = apply_policy(rule, make_catalog(), set())
        assert view.tables == make_catalog().tables
        # one warning with a count (the table, the column and the unknown table), and none of the policy's names:
        # job logs can be streamed to the server
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "3 exclusion(s) match nothing" in message and repr(make_catalog().dataset) in message
        for private in ("patients", "ssn", "visits"):
            assert private not in message

    def test_an_exclusion_that_matches_is_not_reported(self, caplog):
        rule = DatasetRule(share=True, exclude_columns={"person": frozenset({"name"})})
        with caplog.at_level("WARNING", logger="nvflare.app_common.fdn.catalog.site.policy"):
            apply_policy(rule, make_catalog(), set())
        assert not caplog.records

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

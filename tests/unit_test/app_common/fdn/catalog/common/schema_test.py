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

import dataclasses

import pytest

from nvflare.app_common.fdn.catalog.common.schema import (
    CatalogInfo,
    ColumnInfo,
    Lineage,
    Relationship,
    Resource,
    TableInfo,
    ValidationError,
    validate,
)
from tests.unit_test.app_common.fdn.catalog.helpers import LIMITS, make_catalog


class TestWire:
    def test_round_trip(self):
        catalog = make_catalog(observed_at="2026-01-01T00:00:00Z")
        assert CatalogInfo.from_wire(catalog.to_wire()) == catalog

    def test_unknown_key_rejected(self):
        wire = make_catalog().to_wire()
        wire["metadata"]["extra"] = 1
        with pytest.raises(ValidationError):
            CatalogInfo.from_wire(wire)

    def test_missing_key_rejected(self):
        wire = make_catalog().to_wire()
        del wire["metadata"]["owner"]
        with pytest.raises(ValidationError):
            CatalogInfo.from_wire(wire)

    @pytest.mark.parametrize("path", [("metadata", "display_name"), ("dataset",), ("observed_at",)])
    def test_wrong_type_rejected(self, path):
        wire = make_catalog().to_wire()
        target = wire
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = 7
        with pytest.raises(ValidationError):
            CatalogInfo.from_wire(wire)

    def test_error_does_not_echo_source_values(self):
        wire = make_catalog().to_wire()
        wire["metadata"]["tables"][0]["name"] = 12345
        with pytest.raises(ValidationError) as e:
            CatalogInfo.from_wire(wire)
        assert "12345" not in str(e.value)


class TestDigest:
    def test_stable_and_ignores_observed_at(self):
        a = make_catalog(observed_at="2026-01-01T00:00:00Z")
        b = make_catalog(observed_at="2027-01-01T00:00:00Z")
        assert a.digest() == b.digest()

    def test_list_order_is_a_change(self):
        a = make_catalog(modalities=("a", "b"))
        b = make_catalog(modalities=("b", "a"))
        assert a.digest() != b.digest()

    def test_digest_handles_unusual_unicode(self):
        assert make_catalog(display_name="A\ud800 \u00e9").digest()

    def test_metadata_change_changes_digest(self):
        assert make_catalog().digest() != make_catalog(description="x").digest()


class TestValidate:
    def test_valid_catalog_passes(self):
        validate(make_catalog(), LIMITS)

    def test_record_count(self):
        # dataset + 2 tables + 4 columns + 1 relationship + 1 resource + 1 lineage
        assert make_catalog().record_count() == 10

    def test_record_budget(self):
        with pytest.raises(ValidationError):
            validate(make_catalog(), dataclasses.replace(LIMITS, max_records=9))

    def test_duplicate_table(self):
        t = make_catalog().tables[0]
        with pytest.raises(ValidationError):
            validate(make_catalog(tables=(t, t)), LIMITS)

    def test_duplicate_column(self):
        t = TableInfo("t", (ColumnInfo("a", "int"), ColumnInfo("a", "int")))
        with pytest.raises(ValidationError):
            validate(make_catalog(tables=(t,)), LIMITS)

    def test_unknown_primary_key_column(self):
        t = TableInfo("t", (ColumnInfo("a", "int"),), primary_key=("b",))
        with pytest.raises(ValidationError):
            validate(make_catalog(tables=(t,)), LIMITS)

    @pytest.mark.parametrize(
        "rel",
        [
            Relationship(("person_id",), "missing", ("person_id",)),
            Relationship(("person_id",), "person", ("missing",)),
            Relationship(("missing",), "person", ("person_id",)),
            Relationship((), "person", ()),
            Relationship(("person_id",), "person", ("person_id", "name")),
        ],
    )
    def test_bad_relationship(self, rel):
        visit = dataclasses.replace(make_catalog().tables[1], relationships=(rel,))
        with pytest.raises(ValidationError):
            validate(make_catalog(tables=(make_catalog().tables[0], visit)), LIMITS)

    @pytest.mark.parametrize(
        "field,value", [("display_name", ""), ("display_name", "x" * 257), ("description", "x" * 4097)]
    )
    def test_text_bounds(self, field, value):
        with pytest.raises(ValidationError):
            validate(make_catalog(**{field: value}), LIMITS)

    def test_empty_description_is_allowed_but_empty_names_are_not(self):
        validate(make_catalog(description=""), LIMITS)
        with pytest.raises(ValidationError):
            validate(make_catalog(owner=""), LIMITS)

    def test_duplicate_resources_primary_keys_and_lineage(self):
        with pytest.raises(ValidationError):
            validate(make_catalog(resources=(Resource("Patient"), Resource("Patient"))), LIMITS)
        with pytest.raises(ValidationError):
            validate(make_catalog(tables=(TableInfo("t", (ColumnInfo("a", "int"),), primary_key=("a", "a")),)), LIMITS)
        with pytest.raises(ValidationError):
            validate(make_catalog(lineage=(Lineage("raw", "v1"), Lineage("raw", "v2"))), LIMITS)

    def test_description_may_be_long(self):
        validate(make_catalog(description="x" * 4096), LIMITS)

    def test_unsupported_lineage_relation(self):
        with pytest.raises(ValidationError):
            validate(make_catalog(lineage=(Lineage("raw", None, "copied_from"),)), LIMITS)

    def test_duplicate_operations(self):
        with pytest.raises(ValidationError):
            validate(make_catalog(operations=("a.b", "a.b")), LIMITS)

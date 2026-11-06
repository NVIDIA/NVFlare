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
from dataclasses import replace

import pytest

from nvflare.app_common.fdn.catalog.common.payload import ReasonCode, ResultStatus, Selection, build_result
from nvflare.app_common.fdn.catalog.common.schema import ReadLimits, Resource
from nvflare.app_common.fdn.catalog.server.reconcile import Outcome, Reconciler, valid_observation
from nvflare.app_common.fdn.catalog.server.search import SearchQuery
from nvflare.app_common.fdn.catalog.server.store import CatalogStore
from tests.unit_test.app_common.fdn.catalog.helpers import LIMITS, make_catalog, make_large_catalog

NOW = "2026-06-01T00:00:00Z"
STUDY, CLIENT = "study1", "site-a"


@pytest.fixture
def store(tmp_path):
    return CatalogStore(str(tmp_path))


@pytest.fixture
def rec(store):
    return Reconciler(store, LIMITS)


def apply(store, rec, result, now=NOW):
    return rec.apply(STUDY, CLIENT, Selection(None), result, now)


def ok(*catalogs, **kw):
    return build_result(ResultStatus.OK, list(catalogs), **kw)


def row(store, name="clinical-research"):
    return store.client_state(STUDY, CLIENT)["datasets"][name]


class TestApply:
    def test_first_publication(self, store, rec):
        assert apply(store, rec, ok(make_catalog())) == Outcome.APPLIED
        r = row(store)
        assert (r["catalog_version"], r["schema_version"], r["available"], r["withheld"]) == (1, 1, True, False)
        assert r["last_checked_at"] == NOW and r["published_at"] == NOW
        state = store.client_state(STUDY, CLIENT)
        assert state["client"]["health"] == "OK"

    def test_large_metadata_and_withholding_are_applied_together(self, store):
        rec = Reconciler(store, ReadLimits(60, 100, 10000))
        apply(store, rec, ok(make_catalog()))
        catalog = make_large_catalog()
        result = ok(catalog, withheld=["clinical-research"])
        assert len(json.dumps(result, separators=(",", ":"))) > 1_000_000

        assert apply(store, rec, result) == Outcome.APPLIED

        assert row(store)["withheld"] is True
        page = store.search(STUDY, SearchQuery())
        assert page.total == 1 and page.entries[0].catalog == catalog

    def test_unchanged_digest_adds_no_version_but_refreshes_the_check(self, store, rec):
        apply(store, rec, ok(make_catalog()))
        apply(store, rec, ok(make_catalog()), now="2026-06-02T00:00:00Z")
        r = row(store)
        assert r["catalog_version"] == 1 and r["last_checked_at"] == "2026-06-02T00:00:00Z"
        assert r["published_at"] == NOW

    def test_timestamp_only_change_creates_no_version(self, store, rec):
        apply(store, rec, ok(make_catalog(observed_at="2026-05-01T00:00:00Z")))
        apply(store, rec, ok(make_catalog(observed_at="2026-05-02T00:00:00Z")))
        assert row(store)["catalog_version"] == 1
        assert row(store)["observed_at"] == "2026-05-02T00:00:00Z"

    def test_descriptive_change_bumps_catalog_not_schema_version(self, store, rec):
        apply(store, rec, ok(make_catalog()))
        apply(store, rec, ok(make_catalog(description="x")))
        r = row(store)
        assert (r["catalog_version"], r["schema_version"]) == (2, 1)

    def test_structural_change_bumps_both_versions(self, store, rec):
        apply(store, rec, ok(make_catalog()))
        apply(store, rec, ok(make_catalog(resources=(Resource("Observation"),))))
        r = row(store)
        assert (r["catalog_version"], r["schema_version"]) == (2, 2)

    def test_unavailable_keeps_metadata_and_is_restored_by_a_later_read(self, store, rec):
        apply(store, rec, ok(make_catalog()))
        apply(store, rec, ok(unavailable=["clinical-research"]), now="2026-06-02T00:00:00Z")
        r = row(store)
        assert r["available"] is False and r["metadata"] and r["last_checked_at"] == "2026-06-02T00:00:00Z"
        apply(store, rec, ok(make_catalog()))
        assert row(store)["available"] is True

    def test_unavailable_name_never_indexed_stores_nothing(self, store, rec):
        apply(store, rec, ok(unavailable=["never-seen"]))
        assert store.client_state(STUDY, CLIENT)["datasets"] == {}

    def test_withheld_hides_the_entry_everywhere_and_survives_a_bad_catalog(self, store, rec):
        apply(store, rec, ok(make_catalog()))
        bad = make_catalog().to_wire()
        bad["metadata"]["tables"] = "oops"
        result = ok(withheld=["clinical-research"])
        result["catalogs"].append(bad)
        apply(store, rec, result)
        assert row(store)["withheld"] is True
        assert store.search(STUDY, SearchQuery()).total == 0
        apply(store, rec, ok(make_catalog()))
        assert row(store)["withheld"] is False and store.search(STUDY, SearchQuery()).total == 1

    @pytest.mark.parametrize(
        "observed,expected",
        [
            ("2026-05-01T00:00:00Z", "2026-05-01T00:00:00Z"),
            ("garbage", None),
            ("2026-05-01T00:00:00+05:00", None),
            ("2026-05-01 00:00:00Z", None),
            ("2027-01-01T00:00:00Z", None),
            ("2026-05-01T00:00:00", None),
            (None, None),
        ],
    )
    def test_observation_validation(self, store, rec, observed, expected):
        apply(store, rec, ok(make_catalog(observed_at=observed)))
        assert row(store)["observed_at"] == expected

    def test_valid_observation_helper(self):
        assert valid_observation("2026-06-01T00:03:00Z", NOW) == "2026-06-01T00:03:00Z"  # within the skew
        assert valid_observation("2026-06-01T00:06:00Z", NOW) is None

    def test_over_bound_replacement_hides_previous_fields_without_inventing_a_source_check(self, store):
        rec = Reconciler(store, ReadLimits(60, 100, 5))
        previous = make_catalog(tables=(), resources=(), lineage=(), owner="no-longer-public")
        apply(store, rec, ok(previous))
        before = row(store)
        apply(store, rec, ok(make_catalog(owner=None)), now="2026-06-02T00:00:00Z")
        assert row(store) == {**before, "withheld": True}
        assert store.search(STUDY, SearchQuery()).total == 0
        health = store.health(STUDY)[0]
        assert health["health"] == "FAILED" and health["reason_code"] == "INTERNAL"
        assert health["last_success_at"] == NOW and health["last_attempt_at"] == "2026-06-02T00:00:00Z"
        apply(store, rec, ok(replace(previous, owner=None)))
        assert store.search(STUDY, SearchQuery()).entries[0].catalog.owner is None

    @pytest.mark.parametrize("failure", ["malformed", "duplicate"])
    def test_rejected_replacement_does_not_block_healthy_datasets_or_expose_old_metadata(self, store, rec, failure):
        apply(store, rec, ok(make_catalog(owner="private")))
        result = ok(make_catalog(owner=None), make_catalog(dataset="healthy"))
        if failure == "malformed":
            result["catalogs"][0]["metadata"]["tables"] = "bad"
        else:
            result["catalogs"].append(result["catalogs"][0])
        apply(store, rec, result)
        assert row(store)["withheld"] is True
        entries = store.search(STUDY, SearchQuery()).entries
        assert [e.dataset for e in entries] == ["healthy"]
        assert store.health(STUDY)[0]["health"] == "DEGRADED"

    def test_identical_dataset_names_at_two_clients_stay_separate(self, store, rec):
        apply(store, rec, ok(make_catalog()))
        rec.apply(STUDY, "site-b", Selection(None), ok(make_catalog(display_name="Other")), NOW)
        names = {(e.client, e.catalog.display_name) for e in store.search(STUDY, SearchQuery()).entries}
        assert names == {("site-a", "Clinical research"), ("site-b", "Other")}


class TestStatus:
    def test_failed_result_changes_no_availability_and_counts_a_failure(self, store, rec):
        apply(store, rec, ok(make_catalog()))
        failed = build_result(ResultStatus.FAILED, reason_code=ReasonCode.POLICY_UNAVAILABLE, withheld=[])
        apply(store, rec, failed, now="2026-06-02T00:00:00Z")
        state = store.client_state(STUDY, CLIENT)
        assert state["client"]["health"] == "FAILED" and state["client"]["failures"] == 1
        assert state["client"]["reason_code"] == "POLICY_UNAVAILABLE"
        assert state["datasets"]["clinical-research"]["last_checked_at"] == NOW

    def test_budget_reason_degrades_health(self, store, rec):
        apply(store, rec, ok(make_catalog(), reason_code=ReasonCode.BUDGET_EXCEEDED))
        assert store.client_state(STUDY, CLIENT)["client"]["health"] == "DEGRADED"

    def test_unsupported_format_changes_nothing(self, store, rec):
        assert apply(store, rec, {"format_version": 9}) == Outcome.UNSUPPORTED
        assert store.client_state(STUDY, CLIENT)["client"]["health"] == "UNKNOWN"

    def test_no_result_counts_a_failure_and_keeps_availability(self, store, rec):
        apply(store, rec, ok(make_catalog()))
        store.record_no_result(STUDY, CLIENT, "TIMEOUT", NOW)
        state = store.client_state(STUDY, CLIENT)
        assert state["client"]["failures"] == 1 and state["client"]["reason_code"] == "TIMEOUT"
        assert state["client"]["health"] == "FAILED" and row(store)["available"] is True

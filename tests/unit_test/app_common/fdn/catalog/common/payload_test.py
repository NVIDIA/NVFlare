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

from nvflare.app_common.fdn.catalog.common.limits import ReadLimits
from nvflare.app_common.fdn.catalog.common.payload import (
    PayloadError,
    ReasonCode,
    ResultStatus,
    SiteEntry,
    SiteStatus,
    build_catalog_results,
    build_result,
    build_task_payload,
    check_catalog_results,
    parse_result,
    parse_task_payload,
    task_time_budget,
)
from tests.unit_test.app_common.fdn.catalog.helpers import LIMITS, contract_entry


class TestTaskPayload:
    def test_round_trip(self):
        assert parse_task_payload(build_task_payload(["a", "b"])).datasets == ("a", "b")
        assert parse_task_payload(build_task_payload(None)).datasets is None
        assert parse_task_payload(build_task_payload(None, 8.0)).time_budget_s == 8.0
        assert parse_task_payload(build_task_payload(None)).time_budget_s is None

    @pytest.mark.parametrize("timeout", [1, 2, 10, 60, 300, 3600])
    def test_time_budget_leaves_room_to_return_results(self, timeout):
        assert 0 < task_time_budget(timeout) < timeout

    def test_time_already_spent_is_subtracted(self):
        assert task_time_budget(10, 0) == 8
        assert task_time_budget(10, 5) == 3
        assert task_time_budget(10, 30) == 0.1

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            {},
            {"format_version": 1},
            {"format_version": True},
            {"format_version": 2, "datasets": []},
            {"format_version": 2, "datasets": ["a", "a"]},
            {"format_version": 2, "datasets": "a"},
            {"format_version": 2, "previous_datasets": []},
            {"format_version": 2, "time_budget_s": 0},
            {"format_version": 2, "time_budget_s": True},
            {"format_version": 2, "time_budget_s": "8"},
        ],
    )
    def test_invalid(self, payload):
        with pytest.raises(PayloadError):
            parse_task_payload(payload)


class TestParseResult:
    def test_ok_result(self):
        result = build_result(ResultStatus.OK, catalogs=[contract_entry(dataset="a")], unavailable=["b"], skipped=["c"])
        parsed = parse_result(result, None, LIMITS)
        assert parsed.status == ResultStatus.OK
        assert [c["dataset"] for c in parsed.catalogs] == ["a"]
        assert parsed.unavailable == ("b",) and parsed.skipped == ("c",)

    def test_failed_result_keeps_no_catalogs(self):
        result = build_result(ResultStatus.FAILED, reason_code=ReasonCode.CONFIG_UNAVAILABLE)
        result["catalogs"] = [contract_entry(dataset="a")]
        parsed = parse_result(result, None, LIMITS)
        assert parsed.status == ResultStatus.FAILED and parsed.reason_code == ReasonCode.CONFIG_UNAVAILABLE
        assert parsed.catalogs == ()

    def test_unknown_status_is_failed(self):
        result = build_result(ResultStatus.OK)
        result["status"] = "MAYBE"
        assert parse_result(result, None, LIMITS).status == ResultStatus.FAILED

    def test_unsupported_version(self):
        assert parse_result({"format_version": 9}, None, LIMITS).unsupported

    def test_names_outside_the_selection_are_dropped(self):
        result = build_result(
            ResultStatus.OK, catalogs=[contract_entry(dataset="a"), contract_entry(dataset="x")], unavailable=["y"]
        )
        parsed = parse_result(result, ("a",), LIMITS)
        assert [c["dataset"] for c in parsed.catalogs] == ["a"] and parsed.unavailable == ()

    @pytest.mark.parametrize("invalid_schema", [1, b"invalid"])
    def test_malformed_entry_becomes_unavailable_and_others_count(self, invalid_schema):
        bad = contract_entry(dataset="bad")
        bad["contract"]["schema"] = invalid_schema
        result = build_result(ResultStatus.OK, catalogs=[contract_entry(dataset="a")])
        result["catalogs"].append(bad)
        parsed = parse_result(result, None, LIMITS)
        assert [c["dataset"] for c in parsed.catalogs] == ["a"] and parsed.unavailable == ("bad",)

    @pytest.mark.parametrize(
        "field,value",
        [("primaryKeyPosition", "1"), ("primaryKeyPosition", True), ("physicalType", []), ("required", 1), ("id", {})],
    )
    def test_malformed_property_is_isolated_at_receipt(self, field, value):
        bad = contract_entry(dataset="bad")
        bad["contract"]["schema"][0]["properties"][0][field] = value
        parsed = parse_result(build_result(ResultStatus.OK, catalogs=[bad, contract_entry("ok")]), None, LIMITS)
        assert [entry["dataset"] for entry in parsed.catalogs] == ["ok"]
        assert parsed.unavailable == ("bad",)

    def test_server_refuses_reply_over_its_byte_limit(self, monkeypatch):
        from nvflare.app_common.fdn.catalog.common import payload

        reply = build_result(ResultStatus.OK, catalogs=[contract_entry("d")])
        monkeypatch.setattr(payload, "MAX_SITE_REPLY_BYTES", payload.json_size(reply) - 1)
        parsed = parse_result(reply, None, LIMITS)
        assert parsed.status == ResultStatus.FAILED and not parsed.catalogs

    def test_entry_over_the_server_limit_is_skipped(self):
        result = build_result(ResultStatus.OK, catalogs=[contract_entry(dataset="a")])
        parsed = parse_result(result, None, ReadLimits(max_duration_s=60, max_calls=10, max_records=2))
        assert parsed.catalogs == () and parsed.skipped == ("a",) and not parsed.unavailable

    def test_duplicate_dataset_rejects_every_copy(self):
        result = build_result(ResultStatus.OK, catalogs=[contract_entry(dataset="a"), contract_entry(dataset="a")])
        parsed = parse_result(result, None, LIMITS)
        assert parsed.catalogs == () and parsed.unavailable == ("a",)

    def test_lists_are_disjoint(self):
        result = build_result(ResultStatus.OK, catalogs=[contract_entry(dataset="a")], unavailable=["a"], skipped=["a"])
        parsed = parse_result(result, None, LIMITS)
        assert parsed.catalogs == () and parsed.unavailable == ("a",) and parsed.skipped == ()


class TestCatalogResults:
    def test_document(self):
        sites = {
            "site-2": SiteEntry(SiteStatus.NO_RESULT),
            "site-1": SiteEntry(SiteStatus.OK, None, (contract_entry(dataset="a"),), ("b",), ()),
        }
        doc = check_catalog_results(build_catalog_results("oncology", "job-1", "2026-11-09T00:00:00Z", sites))
        assert doc["study"] == "oncology" and doc["job_id"] == "job-1"
        assert list(doc["sites"]) == ["site-1", "site-2"]
        assert doc["sites"]["site-2"] == {"status": "NO_RESULT", "reason_code": None, "unavailable": [], "skipped": []}
        assert [(d["site"], d["dataset"]) for d in doc["datasets"]] == [("site-1", "a")]

    @pytest.mark.parametrize("doc", [None, {}, {"format_version": 1}, {"format_version": 2, "sites": {}}])
    def test_check_rejects_bad_documents(self, doc):
        with pytest.raises(PayloadError):
            check_catalog_results(doc)

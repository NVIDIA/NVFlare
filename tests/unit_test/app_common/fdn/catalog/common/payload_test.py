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

from nvflare.app_common.fdn.catalog.common.payload import (
    PayloadError,
    ReasonCode,
    ResultStatus,
    Selection,
    build_result,
    build_task_payload,
    parse_result,
    parse_task_payload,
)
from tests.unit_test.app_common.fdn.catalog.helpers import make_catalog


class TestTaskPayload:
    def test_round_trip(self):
        payload = parse_task_payload(build_task_payload(["a", "b"], ["a"]))
        assert payload.datasets == ("a", "b") and payload.previous == ("a",)

    def test_omitted_datasets_means_all(self):
        wire = build_task_payload(None, [])
        assert "datasets" not in wire
        assert parse_task_payload(wire).datasets is None

    @pytest.mark.parametrize(
        "bad",
        [
            {"format_version": 2},
            {"format_version": True},
            {"format_version": 1.0},
            {"format_version": 1, "datasets": []},
            {"format_version": 1, "extra": 1},
            {"format_version": 1, "datasets": "a"},
            {"format_version": 1, "datasets": ["a", "a"]},
            {"format_version": 1, "datasets": [""]},
            "not a dict",
        ],
    )
    def test_malformed_rejected(self, bad):
        with pytest.raises(PayloadError):
            parse_task_payload(bad)


class TestParseResult:
    def test_ok_result_round_trip(self):
        result = build_result(ResultStatus.OK, [make_catalog()], withheld=["w"], unavailable=["u"])
        parsed = parse_result(result, Selection(None))
        assert parsed.status == ResultStatus.OK
        assert parsed.catalogs == (make_catalog(),)
        assert parsed.withheld == ("w",) and parsed.unavailable == ("u",)

    @pytest.mark.parametrize("version", [9, True, 1.0, None])
    def test_unsupported_format_version(self, version):
        parsed = parse_result({"format_version": version}, Selection(None))
        assert parsed.unsupported and not parsed.catalogs

    def test_failed_status_never_yields_catalogs_but_keeps_the_restriction_lists(self):
        result = build_result(ResultStatus.FAILED, [make_catalog()], withheld=["w"], unavailable=["u"])
        parsed = parse_result(result, Selection(None))
        assert parsed.status == ResultStatus.FAILED and not parsed.catalogs
        assert parsed.withheld == ("w",) and parsed.unavailable == ("u",)

    def test_names_outside_selection_are_dropped(self):
        result = build_result(ResultStatus.OK, [make_catalog(dataset="other")], withheld=["x"], unavailable=["y"])
        parsed = parse_result(result, Selection(("clinical-research",)))
        assert not parsed.catalogs and not parsed.withheld and not parsed.unavailable

    def test_withheld_wins_over_unavailable_and_catalog(self):
        result = build_result(
            ResultStatus.OK, [make_catalog()], withheld=["clinical-research"], unavailable=["clinical-research"]
        )
        parsed = parse_result(result, Selection(None))
        assert parsed.withheld == ("clinical-research",)
        assert parsed.unavailable == ()
        assert not parsed.catalogs

    def test_unavailable_wins_over_catalog(self):
        result = build_result(ResultStatus.OK, [make_catalog()], unavailable=["clinical-research"])
        parsed = parse_result(result, Selection(None))
        assert parsed.unavailable == ("clinical-research",) and not parsed.catalogs

    def test_malformed_catalog_is_dropped_but_restrictions_survive(self):
        bad = make_catalog(dataset="bad").to_wire()
        bad["metadata"]["tables"] = "oops"
        good = make_catalog(dataset="good")
        result = build_result(ResultStatus.OK, [good], withheld=["w"])
        result["catalogs"].append(bad)
        parsed = parse_result(result, Selection(None))
        assert [c.dataset for c in parsed.catalogs] == ["good"]
        assert parsed.withheld == ("w",)
        assert parsed.rejected == ("bad",)

    def test_duplicate_catalog_entry_drops_every_copy(self):
        result = build_result(ResultStatus.OK, [make_catalog(), make_catalog(), make_catalog(dataset="other")])
        parsed = parse_result(result, Selection(None))
        assert [c.dataset for c in parsed.catalogs] == ["other"]
        assert parsed.rejected == ("clinical-research",)

    def test_malformed_entry_then_valid_entry_for_the_same_name_is_ambiguous(self):
        bad = make_catalog().to_wire()
        bad["metadata"]["tables"] = "oops"
        result = build_result(ResultStatus.OK, [make_catalog()])
        result["catalogs"].insert(0, bad)
        parsed = parse_result(result, Selection(None))
        assert not parsed.catalogs

    def test_build_result_rejects_unknown_status_and_reason(self):
        with pytest.raises(ValueError):
            build_result("NOPE")
        with pytest.raises(ValueError):
            build_result(ResultStatus.FAILED, reason_code="NOPE")

    def test_status_and_reason(self):
        result = build_result(ResultStatus.FAILED, reason_code=ReasonCode.POLICY_UNAVAILABLE)
        parsed = parse_result(result, Selection(None))
        assert parsed.status == ResultStatus.FAILED and parsed.reason_code == ReasonCode.POLICY_UNAVAILABLE

    def test_unknown_status_is_failed(self):
        result = build_result(ResultStatus.OK)
        result["status"] = "WEIRD"
        assert parse_result(result, Selection(None)).status == ResultStatus.FAILED

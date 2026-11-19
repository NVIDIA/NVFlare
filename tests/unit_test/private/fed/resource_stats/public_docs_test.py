# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Keep published resource examples aligned with production readers and rendering."""

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from nvflare.fuel.flare_api.flare_api import Session

if hasattr(Session, "get_job_resources"):
    from dev_tools.generate_resource_stats_examples import examples

ROOT = Path(__file__).resolve().parents[5]
PUBLIC = ROOT / "docs/user_guide/admin_guide/resource_statistics"
GOLDEN = ROOT / "tests/unit_test/private/fed/resource_stats/data/golden/v1"


@pytest.fixture(scope="module")
def schema():
    value = json.loads((ROOT / "docs/schemas/resource_stats.schema.json").read_text())
    Draft202012Validator.check_schema(value)
    return Draft202012Validator(value, format_checker=Draft202012Validator.FORMAT_CHECKER)


@pytest.mark.skipif(not hasattr(Session, "get_job_resources"), reason="requires the resource queries from PR #5640")
def test_published_examples_match_production_generation():
    outputs = examples()
    for name, data in outputs.items():
        assert (GOLDEN / name).read_bytes() == data, name


def test_published_examples_match_golden_files(schema):
    for path in GOLDEN.glob("*.json"):
        schema.validate(json.loads(path.read_bytes()))
    for published, golden in (
        ("participant_summary.json", "participant_summary.json"),
        ("resource_summary.json", "resource_summary.json"),
        ("job_resources.out", "resource_summary.txt"),
        ("study_resources.out", "study_summary.txt"),
    ):
        assert (PUBLIC / "examples" / published).read_bytes() == (GOLDEN / golden).read_bytes()


def test_schema_accepts_zero_memory_limit(schema):
    record = json.loads((PUBLIC / "examples/participant_summary.json").read_bytes())
    record["resource_capacity"]["memory"]["bytes"] = 0
    record["resource_time"]["memory"]["byte_seconds"] = 0
    schema.validate(record)


def test_accepted_entry_requires_original_report_time(schema):
    record = json.loads((PUBLIC / "examples/resource_summary.json").read_bytes())
    del record["participants"][0]["reported_at"]
    assert not schema.is_valid(record)


def _json_blocks(path):
    text = path.read_text()
    for match in re.finditer(r"\.\. code-block:: json\n\n((?:   .*\n|\n)+)", text):
        yield json.loads("\n".join(line[3:] for line in match[1].splitlines()))


@pytest.mark.parametrize("page", ["cpu_work", "message_traffic", "reports"])
def test_json_excerpts_fit_the_public_report(page, schema):
    record = json.loads((PUBLIC / "examples/participant_summary.json").read_bytes())
    blocks = list(_json_blocks(PUBLIC / f"{page}.rst"))
    assert blocks
    for excerpt in blocks:
        value = deepcopy(record)
        utilization = excerpt.pop("resource_utilization", {})
        value.update(excerpt)
        value["resource_utilization"].update(utilization)
        schema.validate(value)


def test_guide_uses_public_interfaces():
    for page in [PUBLIC.with_suffix(".rst"), *PUBLIC.glob("*.rst")]:
        assert not re.search(r"^\s*(?:from|import) nvflare\.private\b", page.read_text(), flags=re.MULTILINE), page

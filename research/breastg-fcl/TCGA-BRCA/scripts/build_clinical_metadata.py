#!/usr/bin/env python3
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

"""Rebuild the case-level clinical TSV from the bundled GDC JSON snapshot.

Select the primary breast diagnosis before reading its stage. A missing or
unknown primary stage stays unknown, even if another malignancy has a stage.
Without an explicit primary breast diagnosis, allow only a unique baseline
breast diagnosis whose classification is missing or not reported. Distinct
competing candidates are errors; array order is never a selection rule.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

DIAGNOSIS_COLUMNS = ("ajcc_pathologic_stage", "tumor_stage", "classification_of_tumor", "primary_diagnosis")
DEMOGRAPHIC_COLUMNS = ("gender", "race", "ethnicity", "vital_status")
COLUMNS = ("case_submitter_id", "case_id") + DIAGNOSIS_COLUMNS + DEMOGRAPHIC_COLUMNS
BREAST_SITES = {
    "breast, nos",
    "nipple",
    "central portion of breast",
    "upper-inner quadrant of breast",
    "lower-inner quadrant of breast",
    "upper-outer quadrant of breast",
    "lower-outer quadrant of breast",
    "axillary tail of breast",
    "overlapping lesion of breast",
}


def _normalize(value):
    return " ".join(str(value or "").strip().lower().split())


def select_diagnosis(case: dict) -> tuple[dict, str]:
    """Return one coherent diagnosis and its selection reason, or an empty dict.

    Exact duplicate diagnoses are harmless. Multiple distinct eligible
    diagnoses require source review instead of silently choosing a stage.
    """
    identity = case.get("submitter_id", "<unknown case>")
    diagnoses = case.get("diagnoses", [])
    if not isinstance(diagnoses, list) or any(not isinstance(diagnosis, dict) for diagnosis in diagnoses):
        raise ValueError(f"{identity}: diagnoses must be a list of objects")
    breast_diagnoses = {
        json.dumps(diagnosis, sort_keys=True): diagnosis
        for diagnosis in diagnoses
        if _normalize(diagnosis.get("tissue_or_organ_of_origin")) in BREAST_SITES
    }
    candidates = [
        diagnosis
        for diagnosis in breast_diagnoses.values()
        if _normalize(diagnosis.get("classification_of_tumor")) == "primary"
    ]
    reason = "primary"
    if not candidates:
        reason = "baseline_breast_fallback"
        candidates = [
            diagnosis
            for diagnosis in breast_diagnoses.values()
            if _normalize(diagnosis.get("classification_of_tumor")) in ("", "not reported")
            and isinstance(diagnosis.get("days_to_diagnosis"), (int, float))
            and not isinstance(diagnosis.get("days_to_diagnosis"), bool)
            and diagnosis["days_to_diagnosis"] == 0
        ]
    if len(candidates) > 1:
        raise ValueError(f"{identity}: ambiguous {reason} diagnoses ({len(candidates)} distinct candidates)")
    return (candidates[0], reason) if candidates else ({}, "no_eligible_diagnosis")


def _identity(case, field):
    value = case.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Clinical case requires a nonempty {field}")
    return value.strip()


def build_clinical_rows(cases: list[dict]) -> list[dict]:
    """Validate case identities and produce rows sorted by case submitter ID."""
    if not isinstance(cases, list):
        raise ValueError("Clinical JSON must be a list of cases")
    rows = []
    seen_submitter_ids, seen_case_ids = set(), set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each clinical case must be an object")
        submitter_id, case_id = _identity(case, "submitter_id"), _identity(case, "case_id")
        if submitter_id in seen_submitter_ids or case_id in seen_case_ids:
            raise ValueError(f"Duplicate clinical case identity: {submitter_id} / {case_id}")
        if "id" in case and case["id"] != case_id:
            raise ValueError(f"{submitter_id}: id does not match case_id")
        seen_submitter_ids.add(submitter_id)
        seen_case_ids.add(case_id)
        diagnosis, _ = select_diagnosis(case)
        demographic = case.get("demographic", {})
        if not isinstance(demographic, dict):
            raise ValueError(f"{submitter_id}: demographic must be an object")
        row = {"case_submitter_id": submitter_id, "case_id": case_id}
        row.update({column: diagnosis.get(column, "") for column in DIAGNOSIS_COLUMNS})
        row.update({column: demographic.get(column, "") for column in DEMOGRAPHIC_COLUMNS})
        rows.append(row)
    return sorted(rows, key=lambda row: row["case_submitter_id"])


def render_clinical_tsv(cases: list[dict]) -> str:
    """Serialize validated rows with the loader's existing schema and LF endings."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(build_clinical_rows(cases))
    return output.getvalue()


def main(argv=None) -> int:
    metadata_dir = Path(__file__).resolve().parents[1] / "metadata"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=metadata_dir / "gdc_clinical_cases.json")
    parser.add_argument("--output", type=Path, default=metadata_dir / "gdc_clinical_cases.tsv")
    parser.add_argument("--check", action="store_true", help="Check exact TSV reproducibility without writing files")
    args = parser.parse_args(argv)
    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError("Input JSON and output TSV must have different paths")
        cases = json.loads(args.input.read_text(encoding="utf-8"))
        rendered = render_clinical_tsv(cases).encode("utf-8")
        if args.check:
            if not args.output.is_file() or args.output.read_bytes() != rendered:
                parser.exit(1, f"Clinical TSV differs from the selected diagnoses: {args.output}\n")
            print(f"Verified {len(cases)} clinical cases: {args.output}")
        else:
            args.output.write_bytes(rendered)
            print(f"Wrote {len(cases)} clinical cases: {args.output}")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

"""Offline clinical diagnosis selection and bundled snapshot regression tests."""

import contextlib
import copy
import csv
import importlib.util
import io
import itertools
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_DIR / "scripts" / "build_clinical_metadata.py"
SPEC = importlib.util.spec_from_file_location("build_clinical_metadata", SCRIPT)
clinical = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clinical)


def diagnosis(classification="primary", stage="Stage IIA", origin="Breast, NOS", days=0):
    return {
        "classification_of_tumor": classification,
        "ajcc_pathologic_stage": stage,
        "tissue_or_organ_of_origin": origin,
        "days_to_diagnosis": days,
        "primary_diagnosis": "Infiltrating duct carcinoma, NOS",
    }


def case(diagnoses, submitter_id="TCGA-TEST-0001", case_id="case-1"):
    return {
        "submitter_id": submitter_id,
        "case_id": case_id,
        "id": case_id,
        "diagnoses": diagnoses,
        "demographic": {"race": "not reported", "vital_status": "Alive"},
    }


class ClinicalDiagnosisSelectionTest(unittest.TestCase):
    def test_primary_selection_is_independent_of_diagnosis_order(self):
        primary = diagnosis(stage="Stage IIIC")
        others = [diagnosis("metastasis", "", "Bone, NOS", 100), diagnosis("Synchronous primary", "Stage I")]
        for diagnoses in itertools.permutations([primary] + others):
            with self.subTest(diagnoses=diagnoses):
                selected, reason = clinical.select_diagnosis(case(list(diagnoses)))
                self.assertEqual(selected, primary)
                self.assertEqual(reason, "primary")

    def test_unknown_primary_stage_is_never_filled_from_another_diagnosis(self):
        for stage in (None, "", "Stage X", "Not Reported"):
            with self.subTest(stage=stage):
                primary = diagnosis(stage=stage)
                diagnoses = [
                    diagnosis("Prior primary", "Stage II"),
                    primary,
                    diagnosis("Synchronous primary", "Stage I"),
                ]
                selected, reason = clinical.select_diagnosis(case(diagnoses))
                self.assertEqual(selected, primary)
                self.assertEqual(reason, "primary")

    def test_nonbreast_primary_is_not_used_for_breast_stage(self):
        primary = diagnosis(stage="Stage IIA")
        kidney = diagnosis(stage="Stage I", origin="Kidney, NOS")
        self.assertEqual(clinical.select_diagnosis(case([kidney, primary])), (primary, "primary"))
        self.assertEqual(clinical.select_diagnosis(case([kidney])), ({}, "no_eligible_diagnosis"))

    def test_unique_unclassified_baseline_breast_diagnosis_is_an_explicit_fallback(self):
        for classification in (None, "", "not reported", " Not   Reported "):
            with self.subTest(classification=classification):
                baseline = diagnosis(classification, "Stage IA", "Upper-inner quadrant of breast")
                self.assertEqual(
                    clinical.select_diagnosis(case([diagnosis("metastasis", "Stage IV"), baseline])),
                    (baseline, "baseline_breast_fallback"),
                )

    def test_known_stage_does_not_make_an_ineligible_fallback_eligible(self):
        candidates = [
            diagnosis("Prior primary"),
            diagnosis("Synchronous primary"),
            diagnosis("Subsequent Primary"),
            diagnosis("recurrence"),
            diagnosis("metastasis"),
            diagnosis("not reported", origin="Kidney, NOS"),
        ]
        candidates.extend(diagnosis("not reported", days=days) for days in (None, 1, -1, "0", False))
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                self.assertEqual(clinical.select_diagnosis(case([candidate])), ({}, "no_eligible_diagnosis"))

    def test_duplicate_diagnoses_are_deduplicated_but_distinct_primary_candidates_fail(self):
        primary = diagnosis()
        self.assertEqual(clinical.select_diagnosis(case([primary, copy.deepcopy(primary)])), (primary, "primary"))
        for other in (diagnosis(stage="Stage IIIA"), {**primary, "year_of_diagnosis": 2003}):
            for diagnoses in ([primary, other], [other, primary]):
                with self.subTest(diagnoses=diagnoses), self.assertRaisesRegex(ValueError, "ambiguous primary"):
                    clinical.select_diagnosis(case(diagnoses))

    def test_multiple_baseline_candidates_fail_instead_of_preferring_known_stage(self):
        candidates = [diagnosis("not reported", ""), diagnosis("not reported", "Stage IA")]
        for diagnoses in (candidates, list(reversed(candidates))):
            with self.assertRaisesRegex(ValueError, "ambiguous baseline_breast_fallback"):
                clinical.select_diagnosis(case(diagnoses))

    def test_export_keeps_diagnosis_fields_together_and_demographics_on_the_case(self):
        primary = diagnosis(stage="Stage IIIC")
        prior = {**diagnosis("Prior primary", "Stage II"), "tumor_stage": "stage ii", "primary_diagnosis": "Other"}
        row = clinical.build_clinical_rows([case([prior, primary])])[0]
        self.assertEqual(row["ajcc_pathologic_stage"], "Stage IIIC")
        self.assertEqual(row["classification_of_tumor"], "primary")
        self.assertEqual(row["primary_diagnosis"], primary["primary_diagnosis"])
        self.assertEqual(row["tumor_stage"], "")
        self.assertEqual(row["vital_status"], "Alive")

    def test_case_order_does_not_affect_serialized_tsv(self):
        first = case([diagnosis()])
        second = case([], "TCGA-TEST-0002", "case-2")
        first_output = clinical.render_clinical_tsv([first, second])
        self.assertEqual(first_output, clinical.render_clinical_tsv([second, first]))
        rows = list(csv.DictReader(io.StringIO(first_output), delimiter="\t"))
        self.assertEqual([row["case_submitter_id"] for row in rows], [first["submitter_id"], second["submitter_id"]])
        self.assertEqual(rows[1]["ajcc_pathologic_stage"], "")

    def test_missing_duplicate_or_conflicting_case_identities_are_rejected(self):
        original = case([diagnosis()])
        for field in ("submitter_id", "case_id"):
            for value in (None, "", " "):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "nonempty"):
                    clinical.build_clinical_rows([{**original, field: value}])
        duplicates = [copy.deepcopy(original), case([], "TCGA-TEST-0002"), case([], case_id="case-2")]
        for duplicate in duplicates:
            with self.subTest(duplicate=duplicate), self.assertRaisesRegex(ValueError, "Duplicate clinical case"):
                clinical.build_clinical_rows([original, duplicate])
        with self.assertRaisesRegex(ValueError, "id does not match"):
            clinical.build_clinical_rows([{**original, "id": "other-case"}])

    def test_invalid_source_structure_is_rejected(self):
        invalid_sources = [{"cases": []}, [None], [case(None)], [case([None])], [{**case([]), "demographic": []}]]
        for source in invalid_sources:
            with self.subTest(source=source), self.assertRaises(ValueError):
                clinical.render_clinical_tsv(source)

    def test_cli_checks_without_writing_and_rebuilds_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "clinical.json", Path(directory) / "clinical.tsv"
            source.write_text(json.dumps([case([diagnosis()])]), encoding="utf-8")
            args = ["--input", str(source), "--output", str(output)]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                clinical.main(args + ["--check"])
            self.assertEqual(error.exception.code, 1)
            self.assertFalse(output.exists())
            output.write_bytes(b"stale clinical table\n")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                clinical.main(args + ["--check"])
            self.assertEqual(error.exception.code, 1)
            self.assertEqual(output.read_bytes(), b"stale clinical table\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(clinical.main(args), 0)
                expected = output.read_bytes()
                self.assertEqual(clinical.main(args + ["--check"]), 0)
                self.assertEqual(clinical.main(args), 0)
            self.assertEqual(output.read_bytes(), expected)

    def test_cli_invalid_source_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "clinical.json", Path(directory) / "clinical.tsv"
            source.write_text(json.dumps([case([diagnosis(), diagnosis(stage="Stage III")])]), encoding="utf-8")
            output.write_bytes(b"keep the existing table\n")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                clinical.main(["--input", str(source), "--output", str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.read_bytes(), b"keep the existing table\n")
            before = source.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                clinical.main(["--input", str(source), "--output", str(source)])
            self.assertEqual(source.read_bytes(), before)


class BundledClinicalMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((PROJECT_DIR / "metadata" / "gdc_clinical_cases.json").read_text(encoding="utf-8"))
        cls.by_patient = {item["submitter_id"]: item for item in cls.cases}

    def test_bundled_tsv_is_exactly_reproducible_from_source_json(self):
        expected = (PROJECT_DIR / "metadata" / "gdc_clinical_cases.tsv").read_bytes()
        self.assertEqual(clinical.render_clinical_tsv(self.cases).encode("utf-8"), expected)
        reversed_cases = [dict(item, diagnoses=list(reversed(item.get("diagnoses", [])))) for item in self.cases]
        self.assertEqual(clinical.render_clinical_tsv(list(reversed(reversed_cases))).encode("utf-8"), expected)

    def test_real_primary_metastatic_and_other_organ_regressions(self):
        for patient, stage in (("TCGA-A2-A04P", "Stage IIIC"), ("TCGA-BH-A0EA", "Stage IIA")):
            with self.subTest(patient=patient):
                selected, reason = clinical.select_diagnosis(self.by_patient[patient])
                self.assertEqual(selected["ajcc_pathologic_stage"], stage)
                self.assertEqual(selected["classification_of_tumor"], "primary")
                self.assertEqual(selected["tissue_or_organ_of_origin"], "Breast, NOS")
                self.assertEqual(reason, "primary")

    def test_real_unknown_primary_is_not_replaced_by_prior_primary_stage(self):
        selected, reason = clinical.select_diagnosis(self.by_patient["TCGA-B6-A0I8"])
        self.assertEqual(selected["ajcc_pathologic_stage"], "Stage X")
        self.assertEqual(selected["classification_of_tumor"], "primary")
        self.assertEqual(reason, "primary")

    def test_real_unclassified_baseline_fallback_and_absent_diagnoses(self):
        selected, reason = clinical.select_diagnosis(self.by_patient["TCGA-A7-A0DC"])
        self.assertEqual(selected["ajcc_pathologic_stage"], "Stage IA")
        self.assertEqual(reason, "baseline_breast_fallback")
        self.assertEqual(clinical.select_diagnosis(self.by_patient["TCGA-BH-A0B2"]), ({}, "no_eligible_diagnosis"))

    def test_snapshot_selection_and_unknown_stage_counts(self):
        selections = [clinical.select_diagnosis(item) for item in self.cases]
        self.assertEqual(len(selections), 1098)
        self.assertEqual(
            Counter(reason for _, reason in selections),
            {"primary": 1096, "baseline_breast_fallback": 1, "no_eligible_diagnosis": 1},
        )
        self.assertEqual(
            sum(selected.get("ajcc_pathologic_stage", "") in ("", "Stage X") for selected, _ in selections), 25
        )


if __name__ == "__main__":
    unittest.main()

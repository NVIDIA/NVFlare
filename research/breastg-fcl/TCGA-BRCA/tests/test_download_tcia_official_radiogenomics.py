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

"""Offline integrity, workbook schema, cohort and generated-table checks."""

import copy
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from scripts import download_tcia_official_radiogenomics as downloader


class TCIADownloadTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.destination = Path(self.directory.name) / "source" / "artifact.xlsx"
        self.payload = b"verified workbook fixture"
        self.digest = hashlib.sha256(self.payload).hexdigest()

    def download(self, **kwargs):
        return downloader.download_verified(
            "https://example.org/artifact.xlsx", self.destination, self.digest, **kwargs
        )

    def response(self, payload):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = payload
        return response

    def test_verified_download_creates_parent_and_persists_bytes(self):
        with patch.object(downloader.urllib.request, "urlopen", return_value=self.response(self.payload)) as request:
            self.assertEqual(self.download(), self.digest)
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assertEqual(request.call_args.kwargs["timeout"], 120)

    def test_valid_existing_artifact_is_reused_without_network(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(self.payload)
        with patch.object(downloader.urllib.request, "urlopen") as request:
            self.assertEqual(self.download(), self.digest)
        request.assert_not_called()

    def test_corrupt_existing_artifact_is_rejected_without_network(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"corrupt")
        with patch.object(downloader.urllib.request, "urlopen") as request:
            with self.assertRaisesRegex(RuntimeError, "wrong SHA-256"):
                self.download()
        request.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), b"corrupt")

    def test_download_checksum_mismatch_removes_invalid_artifact(self):
        with patch.object(downloader.urllib.request, "urlopen", return_value=self.response(b"corrupt")):
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                self.download()
        self.assertFalse(self.destination.exists())

    def test_force_replaces_existing_artifact_with_verified_download(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"corrupt")
        with patch.object(downloader.urllib.request, "urlopen", return_value=self.response(self.payload)) as request:
            self.assertEqual(self.download(force=True), self.digest)
        request.assert_called_once()
        self.assertEqual(self.destination.read_bytes(), self.payload)


class TCIAWorkbookTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.ids = [f"TCGA-ZZ-{index:04d}" for index in range(84)]
        labels = [label.lower() for label, count in downloader.EXPECTED_LABEL_COUNTS.items() for _ in range(count)]
        self.radiomics = pd.DataFrame(
            {
                "Lesion Name": [f"lesion-{patient}" for patient in self.ids],
                "Morphology": list(range(84)),
                **{column: [float(index + 1)] * 84 for index, column in enumerate(downloader.TEMPORAL_COLUMNS)},
            }
        )
        self.pam50 = pd.DataFrame({"CLID": [patient + "-01A" for patient in self.ids], "Pam50.Call": labels})
        self.clinical = pd.DataFrame(
            {
                "bcr_patient_barcode": self.ids,
                "age_at_initial_pathologic_diagnosis": [50] * 84,
                "ajcc_neoplasm_disease_stage": ["Stage I"] * 84,
            }
        )

    def build(self):
        tables = {
            downloader.ARTIFACTS["radiomics"]["filename"]: self.radiomics,
            downloader.ARTIFACTS["pam50"]["filename"]: self.pam50,
            downloader.ARTIFACTS["clinical"]["filename"]: self.clinical,
        }
        # Fixtures exercise conversion without network access or real patient data.
        with patch.object(downloader.pd, "read_excel", side_effect=lambda path, **kwargs: tables[path.name].copy()):
            return downloader.build_tables(self.root / "source", self.root / "output")

    def test_valid_workbooks_join_by_patient_and_separate_feature_columns(self):
        self.pam50 = self.pam50.iloc[::-1].reset_index(drop=True)
        self.clinical = self.clinical.iloc[::-1].reset_index(drop=True)
        # The radiomics workbook may include patients outside the matched cohort.
        extra = self.radiomics.iloc[[0]].copy()
        extra["Lesion Name"] = "TCGA-ZZ-9999"
        self.radiomics = pd.concat([self.radiomics, extra], ignore_index=True)
        summary = self.build()
        self.assertEqual(summary["official_radiomics_rows"], 85)
        self.assertEqual(summary["matched_cohort_rows"], 84)
        self.assertEqual(summary["pam50_label_counts"], downloader.EXPECTED_LABEL_COUNTS)
        self.assertEqual(summary["spatial_feature_dim"], 1)
        self.assertEqual(summary["temporal_feature_dim"], len(downloader.TEMPORAL_COLUMNS))
        output = self.root / "output"
        spatial = pd.read_csv(output / "official_spatial_patient_features.csv")
        temporal = pd.read_csv(output / "official_temporal_patient_features.csv")
        full = pd.read_csv(output / "official_radiomics_patient_features.csv")
        phenotype = pd.read_csv(output / "official_pam50_clinical.tsv", sep="\t")
        self.assertEqual(spatial.columns.tolist(), ["PatientID", "Morphology"])
        self.assertEqual(temporal.columns.tolist(), ["PatientID", *downloader.TEMPORAL_COLUMNS])
        self.assertEqual(full.shape, (84, 2 + len(downloader.TEMPORAL_COLUMNS)))
        self.assertEqual(spatial["PatientID"].tolist(), self.ids)
        self.assertEqual(spatial["Morphology"].tolist(), list(range(84)))
        self.assertEqual(phenotype["_PATIENT"].tolist(), self.ids)
        self.assertEqual(phenotype.loc[0, "PAM50Call_RNAseq"], "Basal")
        self.assertEqual(phenotype["sampleID"].tolist(), [patient + "-01" for patient in self.ids])

    def test_missing_required_workbook_columns_are_rejected(self):
        for table_name, column in (
            ("radiomics", "Lesion Name"),
            ("pam50", "CLID"),
            ("radiomics", downloader.TEMPORAL_COLUMNS[0]),
        ):
            with self.subTest(table=table_name, column=column):
                original = getattr(self, table_name)
                setattr(self, table_name, original.drop(columns=[column]))
                try:
                    with self.assertRaises(RuntimeError):
                        self.build()
                finally:
                    setattr(self, table_name, original)
        self.assertFalse((self.root / "output").exists())

    def test_invalid_or_duplicate_radiomics_identifiers_are_rejected(self):
        for identifier in ("missing", self.radiomics.loc[1, "Lesion Name"]):
            with self.subTest(identifier=identifier):
                self.radiomics.loc[0, "Lesion Name"] = identifier
                with self.assertRaisesRegex(RuntimeError, "identifiers are missing or duplicated"):
                    self.build()
        self.assertFalse((self.root / "output").exists())

    def test_duplicate_join_keys_are_rejected(self):
        for table_name in ("pam50", "clinical"):
            with self.subTest(table=table_name):
                original = getattr(self, table_name)
                setattr(self, table_name, pd.concat([original, original.iloc[[0]]], ignore_index=True))
                try:
                    with self.assertRaises(pd.errors.MergeError):
                        self.build()
                finally:
                    setattr(self, table_name, original)
        self.assertFalse((self.root / "output").exists())

    def test_incorrect_cohort_size_or_label_counts_are_rejected(self):
        original = copy.deepcopy(self.pam50)
        for change in ("missing_patient", "unknown_label", "wrong_label_counts"):
            with self.subTest(change=change):
                self.pam50 = original.copy()
                if change == "missing_patient":
                    self.pam50 = self.pam50.iloc[1:]
                else:
                    self.pam50.loc[0, "Pam50.Call"] = "unknown" if change == "unknown_label" else "luma"
                with self.assertRaisesRegex(RuntimeError, "cohort validation failed"):
                    self.build()
        self.assertFalse((self.root / "output").exists())

    def test_missing_clinical_stage_is_rejected(self):
        self.clinical = self.clinical.iloc[1:]
        with self.assertRaisesRegex(RuntimeError, "missing pathologic stage"):
            self.build()
        self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()

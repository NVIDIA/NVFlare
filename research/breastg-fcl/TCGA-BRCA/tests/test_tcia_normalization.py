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

"""TCIA preprocessing must fit only rows matched by training records."""

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils.tcia_mri_utils import aggregate_mri_features, load_tcia_mri_features, normalize_training_mri_features


class TCIANormalizationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.records = [{"case_submitter_id": f"patient-{index}"} for index in range(1, 4)]

    def table(self, name, rows, header=("PatientID", "f1", "f2"), delimiter=","):
        path = self.root / name
        with path.open("w", newline="") as file:
            writer = csv.writer(file, delimiter=delimiter)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def assert_stats_equal(self, first, second):
        self.assertEqual(first["fit_ids"], second["fit_ids"])
        self.assertEqual(first["n_samples_seen"], second["n_samples_seen"])
        np.testing.assert_array_equal(first["mean"], second["mean"])
        np.testing.assert_array_equal(first["scale"], second["scale"])

    def test_csv_evaluation_and_unrelated_rows_cannot_change_training_fit(self):
        original = self.table(
            "original.csv",
            [("patient-1", 2, 5), ("patient-2", 6, 5), ("patient-3", 10, 20), ("unrelated", -10, 8)],
        )
        changed = self.table(
            "changed.csv",
            [("new-unrelated", 9e8, -9e8), ("patient-3", -1e7, 7e7), ("patient-2", 6, 5), ("patient-1", 2, 5)],
        )
        normalized, audit = normalize_training_mri_features(self.records, [0, 1], load_tcia_mri_features(original))
        modified, other_audit = normalize_training_mri_features(self.records, [0, 1], load_tcia_mri_features(changed))
        self.assert_stats_equal(audit, other_audit)
        self.assertEqual(audit["fit_ids"], ["patient-1", "patient-2"])
        self.assertEqual(audit["n_samples_seen"], 2)
        np.testing.assert_array_equal(audit["mean"], [4.0, 5.0])
        np.testing.assert_array_equal(audit["scale"], [2.0, 1.0])
        for identifier in audit["fit_ids"]:
            np.testing.assert_array_equal(normalized[identifier], modified[identifier])
        np.testing.assert_array_equal(normalized["patient-1"], [-1.0, 0.0])
        np.testing.assert_array_equal(normalized["patient-2"], [1.0, 0.0])
        self.assertFalse(np.array_equal(normalized["patient-3"], modified["patient-3"]))

    def test_npz_evaluation_perturbation_cannot_change_training_fit(self):
        original, changed = self.root / "original.npz", self.root / "changed.npz"
        ids = np.asarray(["patient-1", "patient-2", "patient-3"])
        np.savez(original, case_ids=ids, features=np.asarray([[1, 4], [5, 4], [100, 3]], dtype=np.float64))
        np.savez(changed, case_ids=ids, features=np.asarray([[1, 4], [5, 4], [-1e8, 1e8]], dtype=np.float64))
        normalized, audit = normalize_training_mri_features(self.records, [0, 1], load_tcia_mri_features(original))
        modified, other_audit = normalize_training_mri_features(self.records, [0, 1], load_tcia_mri_features(changed))
        self.assert_stats_equal(audit, other_audit)
        np.testing.assert_array_equal(normalized["patient-1"], modified["patient-1"])
        np.testing.assert_array_equal(normalized["patient-2"], modified["patient-2"])

    def test_matching_is_shared_between_fit_and_aggregation(self):
        ids = ["case-uuid", "patient-id", "sample-id", "sample-submitter", "file-id", "TCGA-AA-0006"]
        records = [
            {"case_id": "case-uuid", "sample_id": "not-selected"},
            {"case_submitter_id": "patient-id"},
            {"sample_id": "sample-id"},
            {"sample_submitter_id": "sample-submitter"},
            {"file_id": "file-id"},
            {"sample_id": "TCGA-AA-0006-01A"},
        ]
        features = {identifier: np.asarray([index, 3.0]) for index, identifier in enumerate(ids)}
        features["not-selected"] = np.asarray([1e8, 1e8])
        normalized, audit = normalize_training_mri_features(records, range(len(records)), features)
        pooled, matched = aggregate_mri_features(records, range(len(records)), normalized)
        self.assertEqual(matched, ids)
        self.assertEqual(audit["fit_ids"], sorted(ids))
        self.assertEqual(audit["n_samples_seen"], len(ids))
        np.testing.assert_allclose(pooled, [0.0, 0.0], atol=1e-7)

    def test_paired_samples_and_duplicate_expression_files_do_not_reweight_fit(self):
        records = [
            {"case_submitter_id": "patient-1", "sample_id": "normal", "file_id": "normal-file"},
            {"case_submitter_id": "patient-1", "sample_id": "tumor", "file_id": "tumor-file-1"},
            {"case_submitter_id": "patient-1", "sample_id": "tumor", "file_id": "tumor-file-2"},
            {"case_submitter_id": "patient-2", "sample_id": "tumor-2", "file_id": "tumor-file-3"},
        ]
        normalized, audit = normalize_training_mri_features(
            records, [0, 1, 2, 3, 0], {"patient-1": [0.0], "patient-2": [10.0]}
        )
        self.assertEqual(audit["n_samples_seen"], 2)
        np.testing.assert_array_equal(audit["mean"], [5.0])
        np.testing.assert_array_equal(audit["scale"], [5.0])
        np.testing.assert_array_equal(normalized["patient-1"], [-1.0])

    def test_no_training_match_never_falls_back_to_evaluation_rows(self):
        for indices in ([0], []):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "No TCIA MRI feature rows match"):
                normalize_training_mri_features(self.records, indices, {"patient-3": [1, 2]})

    def test_constant_columns_and_single_matched_training_patient(self):
        normalized, audit = normalize_training_mri_features(
            self.records, [0], {"patient-1": [3.0, 8.0], "patient-3": [9.0, -2.0]}
        )
        self.assertEqual(audit["n_samples_seen"], 1)
        np.testing.assert_array_equal(audit["scale"], [1.0, 1.0])
        np.testing.assert_array_equal(normalized["patient-1"], [0.0, 0.0])
        np.testing.assert_array_equal(normalized["patient-3"], [6.0, -10.0])
        self.assertEqual(normalized["patient-1"].dtype, np.float32)

    def test_tsv_explicit_id_schema_keeps_feature_positions(self):
        path = self.table("features.tsv", [("patient-1", 2, 30), ("patient-2", 4, 40)], ("subject", "f1", "f2"), "\t")
        loaded = load_tcia_mri_features(path, id_column="subject")
        np.testing.assert_array_equal(loaded["patient-1"], [2.0, 30.0])
        with self.assertRaises(ValueError):
            load_tcia_mri_features(path, id_column="missing")

    def test_missing_or_nonnumeric_evaluation_values_are_rejected_without_column_compression(self):
        for bad in ("", "not-numeric", "nan", "inf"):
            with self.subTest(value=bad):
                path = self.table("invalid.csv", [("patient-1", 2, 30), ("patient-3", bad, 90)])
                with self.assertRaises(ValueError):
                    load_tcia_mri_features(path)
        path = self.table("short.csv", [("patient-1", 2, 30), ("patient-3", 90)])
        with self.assertRaises(ValueError):
            load_tcia_mri_features(path)
        path = self.table("long.csv", [("patient-1", 2, 30), ("patient-3", 1, 2, 3)])
        with self.assertRaises(ValueError):
            load_tcia_mri_features(path)

    def test_duplicate_or_empty_row_ids_are_rejected(self):
        for rows in ([("patient-1", 1, 2), ("patient-1", 3, 4)], [("", 1, 2)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                load_tcia_mri_features(self.table("ids.csv", rows))

    def test_duplicate_or_empty_feature_headers_are_rejected(self):
        for header in (("PatientID", "f", "f"), ("PatientID", "", "f"), ("PatientID",)):
            with self.subTest(header=header), self.assertRaises(ValueError):
                load_tcia_mri_features(self.table("header.csv", [], header))

    def test_npz_id_variants_and_byte_ids_are_loaded_raw(self):
        for key in ("case_ids", "sample_ids", "ids"):
            with self.subTest(key=key):
                path = self.root / f"{key}.npz"
                np.savez(path, features=np.asarray([[4, 9], [8, 7]]), **{key: np.asarray([b"patient-1", b"patient-2"])})
                loaded = load_tcia_mri_features(path)
                np.testing.assert_array_equal(loaded["patient-1"], [4.0, 9.0])

    def test_npz_rejects_bad_shapes_missing_ids_and_nonfinite_values(self):
        invalid = [
            {"features": np.ones((2, 2))},
            {"ids": np.asarray(["patient-1"])},
            {"ids": np.asarray(["patient-1"]), "features": np.ones((2, 2))},
            {"ids": np.asarray([["patient-1"]]), "features": np.ones((1, 2))},
            {"ids": np.asarray(["patient-1"]), "features": np.asarray([1, 2])},
            {"ids": np.asarray(["patient-1"]), "features": np.asarray([[float("nan"), 1]])},
            {"ids": np.asarray(["patient-1", "patient-1"]), "features": np.ones((2, 2))},
        ]
        for arrays in invalid:
            with self.subTest(keys=list(arrays)), self.assertRaises(ValueError):
                path = self.root / "invalid.npz"
                np.savez(path, **arrays)
                load_tcia_mri_features(path)

    def test_global_normalization_switch_is_rejected(self):
        path = self.table("features.csv", [("patient-1", 2, 30), ("patient-2", 4, 40)])
        with self.assertRaisesRegex(ValueError, "whole-table normalization is disabled"):
            load_tcia_mri_features(path, normalize=True)

    def test_invalid_training_indices_and_empty_tables_are_rejected(self):
        for index in (-1, 3, 0.5):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "Invalid training record index"):
                normalize_training_mri_features(self.records, [index], {"patient-1": [2, 3]})
        with self.assertRaisesRegex(ValueError, "nonempty feature table"):
            normalize_training_mri_features(self.records, [0], {})


if __name__ == "__main__":
    unittest.main()

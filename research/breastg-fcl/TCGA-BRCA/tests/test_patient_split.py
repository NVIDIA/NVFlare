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

"""Patient-disjoint TCGA partitions and sparse-cohort regression cases."""

import copy
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils import dataset_utils


def add_patient(records, clinical, patient_index, labels, stage="Stage I", copies=2):
    patient = f"TCGA-ZZ-{patient_index:04d}"
    clinical[patient] = {"ajcc_pathologic_stage": stage}
    for label in labels:
        sample = f"{patient}-{'11A' if label == 0 else '01A'}"
        for copy_index in range(copies):
            records.append(
                {
                    "file_id": f"{sample}-file-{copy_index}",
                    "case_submitter_id": patient,
                    "sample_submitter_id": sample,
                    "sample_id": sample,
                    "sample_type": "Solid Tissue Normal" if label == 0 else "Primary Tumor",
                    "label": label,
                }
            )
    return patient


def cohort_fixture():
    records, clinical = [], {}
    # Include mixed-label patients and duplicate expression files for every
    # biological sample; file-level stratification leaks both identities.
    for patient in range(60):
        add_patient(records, clinical, patient, (0, 1), stage=("Stage I", "Stage II", "Stage III")[patient % 3])
    for patient in range(60, 90):
        add_patient(records, clinical, patient, (0,))
    for patient in range(90, 150):
        add_patient(records, clinical, patient, (1,), stage=("Stage I", "Stage II", "Stage III")[patient % 3])
    return records, np.asarray([record["label"] for record in records], dtype=np.int64), clinical


def flatten_clients(indices):
    return [index for tasks in indices.values() for task in tasks.values() for index in task]


class PatientSplitTest(unittest.TestCase):
    def setUp(self):
        self.records, self.targets, self.clinical = cohort_fixture()
        self.opt = SimpleNamespace(
            num_clients=4,
            num_task=3,
            train_split=0.6,
            seed=42,
            include_unknown_stage=False,
            clinical_path="unused-clinical-fixture.tsv",
        )

    def make_tasks(self, strategy, records=None, targets=None, opt=None, clinical=None):
        records = copy.deepcopy(self.records if records is None else records)
        targets = self.targets if targets is None else targets
        opt = copy.deepcopy(self.opt if opt is None else opt)
        clinical = self.clinical if clinical is None else clinical
        with patch.object(dataset_utils, "_load_clinical_cases", return_value=clinical):
            if strategy == "clinical_stage":
                train, test = dataset_utils._make_clinical_stage_task_indices(records, targets, opt)
            else:
                train, test = dataset_utils._make_random_task_indices(records, targets, opt)
        return train, test, opt

    def assert_disjoint_complete(self, records, train, test, expected=None):
        expected = set(range(len(records))) if expected is None else set(expected)
        self.assertTrue(train)
        self.assertTrue(test)
        self.assertEqual(set(train) | set(test), expected)
        self.assertEqual(len(train), len(set(train)))
        self.assertEqual(len(test), len(set(test)))
        self.assertFalse(set(train) & set(test))
        patients = [{records[index]["case_submitter_id"] for index in split} for split in (train, test)]
        samples = [{records[index]["sample_submitter_id"] for index in split} for split in (train, test)]
        self.assertFalse(patients[0] & patients[1])
        self.assertFalse(samples[0] & samples[1])

    def test_patient_split_keeps_mixed_labels_and_duplicate_sample_files_together(self):
        for seed in (0, 17, 42, 99):
            with self.subTest(seed=seed):
                train, test = dataset_utils._split_patient_indices(self.records, self.targets, 0.6, seed)
                self.assert_disjoint_complete(self.records, train, test)
                audit = dataset_utils._assert_disjoint_patient_sample_ids(self.records, train, test)
                for key, indices, field in (
                    ("train_patient_ids", train, "case_submitter_id"),
                    ("test_patient_ids", test, "case_submitter_id"),
                    ("train_sample_ids", train, "sample_submitter_id"),
                    ("test_sample_ids", test, "sample_submitter_id"),
                ):
                    self.assertEqual(audit[key], sorted({self.records[index][field] for index in indices}))

    def test_final_clinical_and_random_assignments_remain_disjoint_for_multiple_seeds(self):
        for strategy in ("clinical_stage", "random"):
            for seed in (0, 17, 42, 99):
                with self.subTest(strategy=strategy, seed=seed):
                    opt = copy.copy(self.opt)
                    opt.seed = seed
                    train, test, _ = self.make_tasks(strategy, opt=opt)
                    self.assert_disjoint_complete(self.records, flatten_clients(train), flatten_clients(test))
                    for split in (train, test):
                        self.assertEqual(set(split), set(range(opt.num_clients)))
                        for tasks in split.values():
                            self.assertEqual(set(tasks), set(range(opt.num_task)))
                            self.assertTrue(all(tasks.values()))

    def test_same_seed_repeats_partition_and_other_seeds_change_patient_membership(self):
        first = dataset_utils._split_patient_indices(self.records, self.targets, 0.6, 42)
        second = dataset_utils._split_patient_indices(self.records, self.targets, 0.6, 42)
        other = dataset_utils._split_patient_indices(self.records, self.targets, 0.6, 43)
        self.assertEqual(first, second)
        self.assertNotEqual(set(first[0]), set(other[0]))
        for strategy in ("clinical_stage", "random"):
            with self.subTest(strategy=strategy):
                first_train, first_test, _ = self.make_tasks(strategy)
                second_train, second_test, _ = self.make_tasks(strategy)
                self.assertEqual(first_train, second_train)
                self.assertEqual(first_test, second_test)

    def test_stratification_uses_patient_label_profiles_not_expression_file_counts(self):
        train, test = dataset_utils._split_patient_indices(self.records, self.targets, 0.6, 42)
        profiles = {}
        for record in self.records:
            profiles.setdefault(record["case_submitter_id"], set()).add(record["label"])
        for indices, expected in ((train, {(0, 1): 36, (0,): 18, (1,): 36}), (test, {(0, 1): 24, (0,): 12, (1,): 24})):
            patient_ids = {self.records[index]["case_submitter_id"] for index in indices}
            counts = Counter(tuple(sorted(profiles[patient])) for patient in patient_ids)
            self.assertEqual(dict(counts), expected)

    def test_clinical_tumors_stay_in_their_stage_task_on_both_sides(self):
        train, test, _ = self.make_tasks("clinical_stage")
        expected_task = {"Stage I": 0, "Stage II": 1, "Stage III": 2}
        for split in (train, test):
            for tasks in split.values():
                for task, indices in tasks.items():
                    for index in indices:
                        record = self.records[index]
                        if record["label"] == 1:
                            stage = self.clinical[record["case_submitter_id"]]["ajcc_pathologic_stage"]
                            self.assertEqual(task, expected_task[stage])

    def test_unknown_stage_tumors_are_excluded_or_explicitly_assigned_advanced(self):
        clinical = copy.deepcopy(self.clinical)
        unknown_patient = "TCGA-ZZ-0090"
        clinical[unknown_patient]["ajcc_pathologic_stage"] = "Not Reported"
        unknown = {i for i, record in enumerate(self.records) if record["case_submitter_id"] == unknown_patient}
        train, test, opt = self.make_tasks("clinical_stage", clinical=clinical)
        self.assert_disjoint_complete(
            self.records, flatten_clients(train), flatten_clients(test), set(range(len(self.records))) - unknown
        )
        self.assertEqual(opt.excluded_unknown_stage_count, len(unknown))
        opt.include_unknown_stage = True
        train, test, _ = self.make_tasks("clinical_stage", clinical=clinical, opt=opt)
        self.assert_disjoint_complete(self.records, flatten_clients(train), flatten_clients(test))
        assigned_tasks = [
            task
            for split in (train, test)
            for tasks in split.values()
            for task, values in tasks.items()
            if unknown & set(values)
        ]
        self.assertTrue(assigned_tasks)
        self.assertEqual(set(assigned_tasks), {2})

    def test_singleton_label_profile_goes_only_to_training(self):
        records, clinical = [], {}
        singleton = add_patient(records, clinical, 0, (0,), copies=3)
        for patient in range(1, 5):
            add_patient(records, clinical, patient, (1,))
        targets = np.asarray([record["label"] for record in records])
        for seed in (0, 17, 42):
            with self.subTest(seed=seed):
                train, test = dataset_utils._split_patient_indices(records, targets, 0.5, seed)
                self.assert_disjoint_complete(records, train, test)
                singleton_indices = {i for i, record in enumerate(records) if record["case_submitter_id"] == singleton}
                self.assertLessEqual(singleton_indices, set(train))
                self.assertFalse(singleton_indices & set(test))

    def test_a_single_patient_cannot_be_duplicated_to_create_an_evaluation_set(self):
        records, clinical = [], {}
        add_patient(records, clinical, 0, (0, 1), copies=2)
        targets = np.asarray([record["label"] for record in records])
        with self.assertRaises(ValueError):
            dataset_utils._split_patient_indices(records, targets, 0.8, 42)

    def test_missing_patient_or_sample_identity_and_conflicting_barcodes_are_rejected(self):
        for mutation in ("missing_patient", "missing_sample", "conflicting_sample_owner", "barcode_patient_mismatch"):
            with self.subTest(mutation=mutation):
                records = copy.deepcopy(self.records)
                if mutation == "missing_patient":
                    records[0]["case_submitter_id"] = ""
                elif mutation == "missing_sample":
                    records[0].pop("sample_submitter_id")
                    records[0].pop("sample_id")
                elif mutation == "conflicting_sample_owner":
                    records[4]["sample_submitter_id"] = records[0]["sample_submitter_id"]
                    records[4]["sample_id"] = records[0]["sample_id"]
                else:
                    records[0]["sample_submitter_id"] = "TCGA-YY-0000-11A"
                    records[0]["sample_id"] = "TCGA-YY-0000-11A"
                with self.assertRaises(ValueError):
                    dataset_utils._split_patient_indices(records, self.targets, 0.6, 42)

    def test_sample_id_alias_is_accepted_without_falling_back_to_file_id(self):
        records = copy.deepcopy(self.records)
        for record in records:
            record.pop("sample_submitter_id")
        train, test = dataset_utils._split_patient_indices(records, self.targets, 0.6, 42)
        audit = dataset_utils._assert_disjoint_patient_sample_ids(records, train, test)
        self.assertEqual(audit["train_sample_ids"], sorted({records[i]["sample_id"] for i in train}))
        self.assertFalse(set(audit["train_patient_ids"]) & set(audit["test_patient_ids"]))

    def test_explicit_identity_audit_rejects_patient_and_sample_overlap(self):
        # 0/2 are different biological samples from one patient; 0/1 are two
        # expression files from one sample. Distinct record indices are insufficient.
        for train, test in (([0], [2]), ([0], [1])):
            with self.subTest(train=train, test=test), self.assertRaises(ValueError):
                dataset_utils._assert_disjoint_patient_sample_ids(self.records, train, test)
        records = copy.deepcopy(self.records)
        records[4]["sample_submitter_id"] = records[0]["sample_submitter_id"]
        records[4]["sample_id"] = records[0]["sample_id"]
        with self.assertRaises(ValueError):
            dataset_utils._assert_disjoint_patient_sample_ids(records, [0], [4])

    def test_final_validator_rejects_empty_cells_and_duplicate_record_assignments(self):
        train, test, _ = self.make_tasks("random")
        for mutation in ("empty_train", "empty_test", "duplicate_train", "duplicate_test", "overlap"):
            with self.subTest(mutation=mutation):
                changed_train, changed_test = copy.deepcopy(train), copy.deepcopy(test)
                if mutation == "empty_train":
                    changed_train[0][0] = []
                elif mutation == "empty_test":
                    changed_test[0][0] = []
                elif mutation == "duplicate_train":
                    changed_train[0][1].append(changed_train[0][0][0])
                elif mutation == "duplicate_test":
                    changed_test[0][1].append(changed_test[0][0][0])
                else:
                    changed_test[0][0].append(changed_train[0][0][0])
                with self.assertRaises(ValueError):
                    dataset_utils._validate_client_task_split(self.records, changed_train, changed_test, self.opt)

    def test_empty_clinical_task_is_not_filled_with_another_tasks_tumors(self):
        records, clinical = [], {}
        for patient in range(40):
            add_patient(records, clinical, patient, (1,), stage="Stage I")
        targets = np.asarray([record["label"] for record in records])
        with self.assertRaises(ValueError):
            self.make_tasks("clinical_stage", records=records, targets=targets, clinical=clinical)

    def test_stage_missing_only_from_evaluation_is_not_borrowed_from_training(self):
        records, clinical = [], {}
        for patient in range(72):
            add_patient(records, clinical, patient, (1,), stage=("Stage I", "Stage II", "Stage III")[patient // 24])
        targets = np.asarray([record["label"] for record in records])
        # Both partitions contain enough patients overall, but the independently
        # fixed patient split leaves advanced-stage patients only in training.
        test = [index for index, record in enumerate(records) if int(record["case_submitter_id"][-4:]) in range(8)]
        test.extend(
            index for index, record in enumerate(records) if int(record["case_submitter_id"][-4:]) in range(24, 32)
        )
        train = sorted(set(range(len(records))) - set(test))
        with patch.object(dataset_utils, "_split_patient_indices", return_value=(train, test)) as split:
            with self.assertRaises(ValueError):
                self.make_tasks("clinical_stage", records=records, targets=targets, clinical=clinical)
            split.assert_called_once()

    def test_insufficient_records_for_client_task_cells_are_not_duplicated(self):
        records, clinical = [], {}
        for patient in range(2):
            add_patient(records, clinical, patient, (0, 1), copies=1)
        targets = np.asarray([record["label"] for record in records])
        for strategy in ("clinical_stage", "random"):
            with self.subTest(strategy=strategy), self.assertRaises(ValueError):
                self.make_tasks(strategy, records=records, targets=targets, clinical=clinical)


class ExpressionCacheIdentityTest(unittest.TestCase):
    def setUp(self):
        records, _, _ = cohort_fixture()
        self.records = [copy.deepcopy(records[index]) for index in (0, 4, 8)]
        source = tempfile.TemporaryDirectory()
        self.addCleanup(source.cleanup)
        for index, record in enumerate(self.records):
            record["file_path"] = str(Path(source.name) / f"expression-{index}.tsv")
            Path(record["file_path"]).write_text(
                "gene_id\tgene_type\ttpm_unstranded\n"
                f"gene-a\tprotein_coding\t{index + 1}\n"
                f"gene-b\tprotein_coding\t{(index + 2) ** 2}\n"
            )
        self.vectors = {
            record["file_path"]: np.asarray([index + 1, (index + 2) ** 2], dtype=np.float32)
            for index, record in enumerate(self.records)
        }
        self.genes = ["gene-a", "gene-b"]

    def read_expression(self, path, _value_column, _max_genes):
        return self.vectors[path], self.genes

    def test_same_ordered_record_identities_reuse_cached_expression(self):
        with tempfile.TemporaryDirectory() as directory:
            opt = SimpleNamespace(data_dir=directory, expression_value_col="tpm_unstranded", max_genes=2)
            with patch.object(dataset_utils, "_read_expression_vector", side_effect=self.read_expression) as read:
                expected = dataset_utils._build_expression_cache(opt, self.records)
                self.assertEqual(read.call_count, len(self.records))
                np.testing.assert_array_equal(expected[0], np.log1p(np.vstack(list(self.vectors.values()))))
                read.reset_mock()
                actual = dataset_utils._build_expression_cache(opt, self.records)
                read.assert_not_called()
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            self.assertEqual(actual[2:], expected[2:])

    def test_same_count_changed_order_file_patient_sample_or_label_rebuilds_cache(self):
        for mutation in ("order", "file", "patient", "sample", "label"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                opt = SimpleNamespace(data_dir=directory, expression_value_col="tpm_unstranded", max_genes=2)
                with patch.object(dataset_utils, "_read_expression_vector", side_effect=self.read_expression) as read:
                    baseline = dataset_utils._build_expression_cache(opt, self.records)
                    changed = copy.deepcopy(self.records)
                    if mutation == "order":
                        changed.reverse()
                    elif mutation == "file":
                        changed[0]["file_id"] = "replacement-file"
                    elif mutation == "patient":
                        changed[0]["case_submitter_id"] = "TCGA-ZZ-9000"
                        changed[0]["sample_submitter_id"] = changed[0]["sample_id"] = "TCGA-ZZ-9000-11A"
                    elif mutation == "sample":
                        changed[0]["sample_submitter_id"] = changed[0]["sample_id"] = "TCGA-ZZ-0000-11B"
                    else:
                        changed[0]["label"] = 1
                    read.reset_mock()
                    actual = dataset_utils._build_expression_cache(opt, changed)
                    self.assertEqual(read.call_count, len(changed))
                self.assertEqual(len(list((Path(directory) / "cache").glob("*.npz"))), 2)
                np.testing.assert_array_equal(actual[1], [record["label"] for record in changed])
                self.assertEqual(actual[2], [record["sample_id"] for record in changed])
                if mutation == "order":
                    np.testing.assert_allclose(actual[0], baseline[0][::-1], rtol=1e-6, atol=1e-6)

    def test_legacy_count_only_cache_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            opt = SimpleNamespace(data_dir=directory, expression_value_col="tpm_unstranded", max_genes=2)
            cache = Path(directory) / "cache"
            cache.mkdir()
            np.savez_compressed(
                cache / "tcga_brca_tpm_unstranded_2_3samples.npz",
                features=np.full((3, 2), 12345.0),
                targets=np.ones(3),
                sample_ids=np.asarray(["wrong-sample"] * 3, dtype=object),
                gene_ids=np.asarray(self.genes, dtype=object),
            )
            with patch.object(dataset_utils, "_read_expression_vector", side_effect=self.read_expression) as read:
                features, targets, sample_ids, _ = dataset_utils._build_expression_cache(opt, self.records)
            self.assertEqual(read.call_count, len(self.records))
            self.assertFalse(np.any(features == 12345.0))
            np.testing.assert_array_equal(targets, [record["label"] for record in self.records])
            self.assertEqual(sample_ids, [record["sample_id"] for record in self.records])

    def test_setup_rejects_an_invalid_patient_partition_before_reading_expression_or_tcia(self):
        with tempfile.TemporaryDirectory() as directory:
            opt = SimpleNamespace(
                output_dir=directory,
                data_dir=directory,
                task_split_strategy="random",
                num_clients=4,
                num_task=3,
                train_split=0.6,
                seed=42,
            )
            with (
                patch.object(dataset_utils, "_read_manifest", return_value=self.records),
                patch.object(dataset_utils, "_build_expression_cache") as cache,
                patch.object(dataset_utils, "load_tcia_mri_features") as tcia,
            ):
                with self.assertRaises(ValueError):
                    dataset_utils.setup_tcga_brca_loaders(opt)
                cache.assert_not_called()
                tcia.assert_not_called()

    def test_setup_rejects_cached_labels_or_samples_that_do_not_match_manifest(self):
        records, targets, _ = cohort_fixture()
        for mutation in ("labels", "samples"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                opt = SimpleNamespace(
                    output_dir=directory,
                    data_dir=directory,
                    task_split_strategy="random",
                    num_clients=4,
                    num_task=3,
                    train_split=0.6,
                    seed=42,
                )
                cached_targets = targets.copy()
                samples = [record["sample_id"] for record in records]
                if mutation == "labels":
                    cached_targets[0] = 1 - cached_targets[0]
                else:
                    samples[0] = "TCGA-ZZ-9000-11A"
                cached = (np.ones((len(records), 2)), cached_targets, samples, self.genes)
                with (
                    patch.object(dataset_utils, "_read_manifest", return_value=records),
                    patch.object(dataset_utils, "_build_expression_cache", return_value=cached),
                    patch.object(dataset_utils, "load_tcia_mri_features") as tcia,
                    patch.object(dataset_utils, "DataLoader") as loader,
                ):
                    with self.assertRaisesRegex(ValueError, "cache.*identities"):
                        dataset_utils.setup_tcga_brca_loaders(opt)
                    tcia.assert_not_called()
                    loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()

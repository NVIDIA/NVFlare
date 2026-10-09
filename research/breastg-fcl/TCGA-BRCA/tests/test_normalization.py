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

"""Training-only expression normalization and loader-level leakage regressions."""

import copy
import csv
import hashlib
import io
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model.modules import BreastGraphGenerator
from utils import dataset_utils
from utils.normalization_utils import apply_standardizer, fit_standardizer


def assigned_indices(partition, split):
    return sorted(
        index
        for tasks in partition[f"client_task_{split}_indices"].values()
        for indices in tasks.values()
        for index in indices
    )


class StandardizerTest(unittest.TestCase):
    def test_float64_population_statistics_and_constant_columns_are_stable(self):
        values = np.asarray([[1, 7, 1e9], [3, 7, 1e9 + 4]], dtype=np.float64)
        original = values.copy()
        stats = fit_standardizer(values)
        self.assertEqual(stats["mean"].dtype, np.float64)
        self.assertEqual(stats["scale"].dtype, np.float64)
        self.assertEqual(stats["n_samples_seen"], 2)
        np.testing.assert_array_equal(stats["mean"], [2, 7, 1e9 + 2])
        np.testing.assert_array_equal(stats["scale"], [1, 1, 2])
        transformed = apply_standardizer(values, stats)
        self.assertEqual(transformed.dtype, np.float32)
        np.testing.assert_array_equal(transformed, [[-1, 0, -1], [1, 0, 1]])
        np.testing.assert_array_equal(values, original)

    def test_evaluation_values_use_frozen_training_statistics(self):
        stats = fit_standardizer(np.asarray([[1, 5], [3, 5]], dtype=np.float32))
        original = copy.deepcopy(stats)
        actual = apply_standardizer(np.asarray([[100, 9], [-50, 1]], dtype=np.float32), stats)
        np.testing.assert_array_equal(actual, [[98, 4], [-52, -4]])
        for field in ("mean", "scale"):
            np.testing.assert_array_equal(stats[field], original[field])
        self.assertEqual(stats["n_samples_seen"], original["n_samples_seen"])

    def test_fitting_rejects_empty_nonmatrix_and_nonfinite_inputs(self):
        for values in ([], [1, 2], np.empty((0, 2)), [[np.nan, 1]], [[1, np.inf]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                fit_standardizer(values)


class TrainingNormalizationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.records, self.clinical = [], {}
        for patient_index in range(54):
            patient = f"TCGA-ZZ-{patient_index:04d}"
            stage = ("Stage I", "Stage II", "Stage III")[patient_index // 16] if patient_index < 48 else "Not Reported"
            self.clinical[patient] = {"ajcc_pathologic_stage": stage}
            for label in ((0, 1) if patient_index < 48 else (1,)):
                sample = f"{patient}-{'11A' if label == 0 else '01A'}"
                self.records.append(
                    {
                        "file_id": f"file-{len(self.records)}",
                        "file_path": str(self.root / f"expression-{len(self.records)}.tsv"),
                        "case_submitter_id": patient,
                        "sample_submitter_id": sample,
                        "sample_id": sample,
                        "sample_type": "Solid Tissue Normal" if label == 0 else "Primary Tumor",
                        "label": label,
                    }
                )
        self.raw = np.asarray([[i + 1, (i * 7) % 13 + 2, 9] for i in range(len(self.records))], dtype=np.float32)
        self.path_to_index = {record["file_path"]: index for index, record in enumerate(self.records)}
        self.genes = ["gene-a", "gene-b", "constant-gene"]
        self.spatial_path = self.root / "spatial.csv"
        self.temporal_path = self.root / "temporal.csv"
        for path, multiplier in ((self.spatial_path, 1), (self.temporal_path, 2)):
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["case_submitter_id", "feature_a", "feature_b"])
                for i, patient in enumerate(self.clinical):
                    writer.writerow([patient, multiplier * (i + 1), (i % 7) ** 2 + 3])
        self.opt = SimpleNamespace(
            num_clients=4,
            num_task=3,
            train_split=0.6,
            seed=42,
            include_unknown_stage=False,
            task_split_strategy="clinical_stage",
            clinical_path="unused-clinical-fixture.tsv",
            expression_value_col="tpm_unstranded",
            max_genes=3,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            tcia_mri_features_path=str(self.spatial_path),
            tcia_dce_kinetics_path=str(self.temporal_path),
            temporal_window=2,
            attention_temperature=1.0,
            graph_epsilon=1e-8,
            gat_hidden_dim=16,
            gat_embedding_dim=8,
            gat_heads=2,
            gat_dropout=0.0,
        )

    def run_loaders(self, name, raw=None, seed=42, cache_name=None):
        opt = copy.deepcopy(self.opt)
        opt.seed = seed
        opt.output_dir = str(self.root / name)
        opt.data_dir = str(self.root / (cache_name or name) / "data")
        raw = self.raw if raw is None else raw
        for index, record in enumerate(self.records):
            Path(record["file_path"]).write_text(
                "gene_id\tgene_type\ttpm_unstranded\n"
                + "".join(f"{gene}\tprotein_coding\t{value}\n" for gene, value in zip(self.genes, raw[index]))
            )

        def read(path, _value_column, _max_genes):
            return raw[self.path_to_index[path]], self.genes

        with (
            patch.object(dataset_utils, "_read_manifest", return_value=copy.deepcopy(self.records)),
            patch.object(dataset_utils, "_load_clinical_cases", return_value=self.clinical),
            patch.object(dataset_utils, "_read_expression_vector", side_effect=read) as reader,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            loaders = dataset_utils.setup_tcga_brca_loaders(opt)
        with (Path(opt.output_dir) / f"tcga_brca_partitioning_seed{seed}.pkl").open("rb") as stream:
            partition = pickle.load(stream)
        features = loaders[0][0]["train"].dataset.dataset.features.numpy().copy()
        return opt, features, partition, reader.call_count

    def graph_sequence(self, opt):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            generator = BreastGraphGenerator(opt)
            return [
                generator.learn(
                    0,
                    task_id=task,
                    spatial_features=opt.client_spatial_features[task],
                    temporal_features=opt.client_temporal_features[task],
                )
                for task in range(opt.num_task)
            ]

    def test_saved_expression_parameters_fit_only_actual_training_rows_and_replay_transform(self):
        _opt, actual, partition, reads = self.run_loaders("baseline")
        train = assigned_indices(partition, "train")
        test = assigned_indices(partition, "test")
        excluded = sorted(set(range(len(self.records))) - set(train) - set(test))
        self.assertEqual(len(excluded), 6)
        self.assertEqual(reads, len(self.records))
        self.assertEqual(partition["normalization"]["schema_version"], 1)
        stats = partition["normalization"]["expression"]
        self.assertEqual(stats["fit_indices"], train)
        self.assertEqual(stats["n_samples_seen"], len(train))
        logged = np.log1p(self.raw)
        np.testing.assert_allclose(stats["mean"], logged[train].astype(np.float64).mean(axis=0), rtol=0, atol=0)
        expected_scale = logged[train].astype(np.float64).std(axis=0)
        expected_scale[expected_scale < 1e-6] = 1.0
        np.testing.assert_array_equal(stats["scale"], expected_scale)
        np.testing.assert_array_equal(apply_standardizer(logged, stats), actual)
        np.testing.assert_allclose(actual[train].mean(axis=0), 0, atol=2e-7)
        np.testing.assert_array_equal(actual[:, 2], np.zeros(len(actual)))
        for modality in ("tcia_spatial", "tcia_temporal"):
            self.assertTrue(partition["normalization"][modality]["fit_ids"])
            self.assertEqual(
                partition["normalization"][modality]["n_samples_seen"],
                len(partition["normalization"][modality]["fit_ids"]),
            )

    def test_changing_evaluation_and_excluded_expression_cannot_change_training_or_graphs(self):
        baseline_opt, baseline, partition, _ = self.run_loaders("baseline")
        train = assigned_indices(partition, "train")
        test = assigned_indices(partition, "test")
        excluded = sorted(set(range(len(self.records))) - set(train) - set(test))
        for label, changed_indices in (("evaluation", test), ("excluded", excluded)):
            with self.subTest(changed=label):
                changed_raw = self.raw.copy()
                changed_raw[changed_indices] = changed_raw[changed_indices] * 1000 + 50000
                changed_opt, changed, changed_partition, _ = self.run_loaders(label, raw=changed_raw)
                self.assertEqual(assigned_indices(changed_partition, "train"), train)
                np.testing.assert_array_equal(changed[train], baseline[train])
                for key in ("mean", "scale"):
                    np.testing.assert_array_equal(
                        changed_partition["normalization"]["expression"][key],
                        partition["normalization"]["expression"][key],
                    )
                self.assertFalse(np.array_equal(changed[changed_indices], baseline[changed_indices]))
                for attribute in ("client_spatial_features", "client_temporal_features"):
                    for first, second in zip(getattr(baseline_opt, attribute), getattr(changed_opt, attribute)):
                        np.testing.assert_array_equal(first, second)
                for first, second in zip(self.graph_sequence(baseline_opt), self.graph_sequence(changed_opt)):
                    np.testing.assert_array_equal(first, second)

    def test_changing_nontraining_tcia_patients_cannot_change_training_summaries_or_graphs(self):
        baseline_opt, baseline, partition, _ = self.run_loaders("baseline", cache_name="shared")
        train = assigned_indices(partition, "train")
        training_patients = {self.records[index]["case_submitter_id"] for index in train}
        changed_rows = 0
        for path in (self.spatial_path, self.temporal_path):
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            for row in rows:
                if row["case_submitter_id"] not in training_patients:
                    row["feature_a"] = str(float(row["feature_a"]) * 1000 + 50000)
                    row["feature_b"] = str(float(row["feature_b"]) * 2000 + 100000)
                    changed_rows += 1
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        self.assertGreater(changed_rows, 0)
        changed_opt, changed, changed_partition, reads = self.run_loaders("perturbed-tcia", cache_name="shared")
        self.assertEqual(reads, 0)
        np.testing.assert_array_equal(changed, baseline)
        for modality in ("tcia_spatial", "tcia_temporal"):
            original = partition["normalization"][modality]
            actual = changed_partition["normalization"][modality]
            self.assertEqual(actual["fit_ids"], original["fit_ids"])
            for key in ("mean", "scale"):
                np.testing.assert_array_equal(actual[key], original[key])
        for attribute in ("client_spatial_features", "client_temporal_features"):
            for original, actual in zip(getattr(baseline_opt, attribute), getattr(changed_opt, attribute)):
                np.testing.assert_array_equal(actual, original)
        for original, actual in zip(self.graph_sequence(baseline_opt), self.graph_sequence(changed_opt)):
            np.testing.assert_array_equal(actual, original)

    def test_new_partition_refits_statistics_while_reusing_only_raw_log1p_cache(self):
        first_opt, first, first_partition, first_reads = self.run_loaders("seed42", seed=42, cache_name="shared")
        _second_opt, second, second_partition, second_reads = self.run_loaders("seed43", seed=43, cache_name="shared")
        first_train, second_train = assigned_indices(first_partition, "train"), assigned_indices(
            second_partition, "train"
        )
        self.assertNotEqual(first_train, second_train)
        self.assertEqual(first_reads, len(self.records))
        self.assertEqual(second_reads, 0)
        self.assertEqual(len(list((Path(first_opt.data_dir) / "cache").glob("*.npz"))), 1)
        logged = np.log1p(self.raw)
        for features, partition, indices in (
            (first, first_partition, first_train),
            (second, second_partition, second_train),
        ):
            stats = partition["normalization"]["expression"]
            self.assertEqual(stats["fit_indices"], indices)
            np.testing.assert_array_equal(stats["mean"], logged[indices].astype(np.float64).mean(axis=0))
            np.testing.assert_array_equal(apply_standardizer(logged, stats), features)
        self.assertFalse(
            np.array_equal(
                first_partition["normalization"]["expression"]["mean"],
                second_partition["normalization"]["expression"]["mean"],
            )
        )

    def test_raw_cache_carries_log1p_schema_and_ignores_previous_standardized_cache(self):
        data_dir = self.root / "legacy" / "data"
        cache = data_dir / "cache"
        cache.mkdir(parents=True)
        identities = [
            (r["file_id"], r["case_submitter_id"], r["sample_submitter_id"], int(r["label"])) for r in self.records
        ]
        fingerprint = hashlib.sha256(json.dumps(identities).encode("utf-8")).hexdigest()
        old = cache / f"tcga_brca_tpm_unstranded_3_{len(self.records)}samples_{fingerprint[:16]}.npz"
        np.savez_compressed(
            old,
            features=np.full(self.raw.shape, -777, dtype=np.float32),
            targets=np.asarray([r["label"] for r in self.records]),
            sample_ids=np.asarray([r["sample_id"] for r in self.records], dtype=object),
            gene_ids=np.asarray(self.genes, dtype=object),
            record_fingerprint=fingerprint,
        )
        _opt, features, _partition, reads = self.run_loaders("with-legacy", cache_name="legacy")
        self.assertEqual(reads, len(self.records))
        self.assertFalse(np.any(features == -777))
        current = list(cache.glob("tcga_brca_log1p_v3_*.npz"))
        self.assertEqual(len(current), 1)
        with np.load(current[0], allow_pickle=False) as saved:
            self.assertEqual(saved["cache_schema_version"].item(), 3)
            self.assertEqual(saved["preprocessing"].item(), "log1p_only")
            np.testing.assert_array_equal(saved["features"], np.log1p(self.raw))


if __name__ == "__main__":
    unittest.main()

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

"""Expression caches are bound to ordered records and actual source bytes."""

import copy
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils import dataset_utils


class ExpressionCacheTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.opt = SimpleNamespace(data_dir=str(self.root / "data"), expression_value_col="tpm_unstranded", max_genes=2)
        self.records = []
        self.raw = np.asarray([[1, 4], [3, 6], [5, 8]], dtype=np.float32)
        for index, values in enumerate(self.raw):
            patient = f"TCGA-AA-{index:04d}"
            label = index % 2
            sample = f"{patient}-{'11A' if label == 0 else '01A'}"
            path = self.root / f"source-{index}.tsv"
            self.write_source(path, values)
            self.records.append(
                {
                    "file_id": f"file-{index}",
                    "file_path": str(path),
                    "case_submitter_id": patient,
                    "sample_submitter_id": sample,
                    "sample_id": sample,
                    "label": label,
                }
            )

    def write_source(self, path, values, genes=("gene-a", "gene-b")):
        Path(path).write_text(
            "gene_id\tgene_type\ttpm_unstranded\n"
            + "".join(f"{gene}\tprotein_coding\t{int(value)}\n" for gene, value in zip(genes, values))
        )

    def cache_files(self):
        return sorted((Path(self.opt.data_dir) / "cache").glob("tcga_brca_log1p_v3_*.npz"))

    def read_cached_fields(self):
        files = self.cache_files()
        self.assertEqual(len(files), 1)
        with np.load(files[0], allow_pickle=False) as saved:
            return files[0], {key: saved[key].copy() for key in saved.files}

    def test_same_inputs_hit_cache_and_store_safe_complete_source_identities(self):
        with patch.object(
            dataset_utils, "_read_expression_vector", wraps=dataset_utils._read_expression_vector
        ) as read:
            first = dataset_utils._build_expression_cache(self.opt, self.records)
            self.assertEqual(read.call_count, len(self.records))
            read.reset_mock()
            second = dataset_utils._build_expression_cache(self.opt, self.records)
            read.assert_not_called()
        np.testing.assert_array_equal(first[0], np.log1p(self.raw))
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2:], second[2:])
        _path, saved = self.read_cached_fields()
        self.assertTrue(all(value.dtype.kind != "O" for value in saved.values()))
        for field, expected in (
            ("file_ids", [r["file_id"] for r in self.records]),
            ("patient_ids", [r["case_submitter_id"] for r in self.records]),
            ("sample_submitter_ids", [r["sample_submitter_id"] for r in self.records]),
            ("sample_ids", [r["sample_id"] for r in self.records]),
            ("targets", [r["label"] for r in self.records]),
            ("source_sha256", [hashlib.sha256(Path(r["file_path"]).read_bytes()).hexdigest() for r in self.records]),
        ):
            np.testing.assert_array_equal(saved[field], expected)
        self.assertEqual(saved["targets"].dtype, np.int64)
        self.assertEqual(saved["features"].dtype, np.float32)
        self.assertEqual(saved["cache_schema_version"].item(), 3)
        self.assertEqual(saved["preprocessing"].item(), "log1p_only")
        self.assertEqual(saved["expression_value_col"].item(), "tpm_unstranded")
        self.assertEqual(saved["max_genes"].item(), 2)
        self.assertEqual(len(saved["record_fingerprint"].item()), 64)

    def test_exact_sample_id_is_checked_even_when_canonical_sample_identity_is_unchanged(self):
        dataset_utils._build_expression_cache(self.opt, self.records)
        changed = copy.deepcopy(self.records)
        changed[0]["sample_id"] = " " + changed[0]["sample_id"] + " "
        with patch.object(
            dataset_utils, "_read_expression_vector", wraps=dataset_utils._read_expression_vector
        ) as read:
            features, targets, sample_ids, _ = dataset_utils._build_expression_cache(self.opt, changed)
        self.assertEqual(read.call_count, len(changed))
        self.assertEqual(len(self.cache_files()), 2)
        self.assertEqual(sample_ids, [r["sample_id"] for r in changed])
        np.testing.assert_array_equal(features, np.log1p(self.raw))
        np.testing.assert_array_equal(targets, [r["label"] for r in changed])

    def test_same_id_same_size_and_restored_mtime_content_change_invalidates_cache(self):
        first = dataset_utils._build_expression_cache(self.opt, self.records)
        source = Path(self.records[0]["file_path"])
        original_stat = source.stat()
        original_size = source.stat().st_size
        self.write_source(source, [9, 4])
        os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        self.assertEqual(source.stat().st_size, original_size)
        self.assertEqual(source.stat().st_mtime_ns, original_stat.st_mtime_ns)
        with patch.object(
            dataset_utils, "_read_expression_vector", wraps=dataset_utils._read_expression_vector
        ) as read:
            actual = dataset_utils._build_expression_cache(self.opt, self.records)
        self.assertEqual(read.call_count, len(self.records))
        self.assertEqual(len(self.cache_files()), 2)
        self.assertNotEqual(actual[0][0, 0], first[0][0, 0])
        np.testing.assert_array_equal(actual[0][0], np.log1p(np.asarray([9, 4], dtype=np.float32)))
        np.testing.assert_array_equal(actual[0][1:], first[0][1:])

    def test_each_persisted_identity_field_is_checked_before_cache_use(self):
        dataset_utils._build_expression_cache(self.opt, self.records)
        path, original = self.read_cached_fields()
        for field in (
            "file_ids",
            "patient_ids",
            "sample_submitter_ids",
            "sample_ids",
            "targets",
            "source_sha256",
            "expression_value_col",
            "max_genes",
            "cache_schema_version",
            "preprocessing",
            "record_fingerprint",
        ):
            with self.subTest(field=field):
                changed = {key: value.copy() for key, value in original.items()}
                if field in ("targets", "max_genes", "cache_schema_version"):
                    changed[field] = changed[field] + 1
                elif changed[field].ndim == 0:
                    changed[field] = np.asarray("incorrect-value")
                else:
                    changed[field] = np.asarray(["incorrect-value", *changed[field].tolist()[1:]])
                np.savez_compressed(path, **changed)
                with patch.object(dataset_utils, "_read_expression_vector") as read:
                    with self.assertRaisesRegex(ValueError, "Invalid expression cache"):
                        dataset_utils._build_expression_cache(self.opt, self.records)
                    read.assert_not_called()
        np.savez_compressed(path, **original)

    def test_missing_fields_wrong_shapes_dtypes_and_nonfinite_cache_are_rejected(self):
        dataset_utils._build_expression_cache(self.opt, self.records)
        path, original = self.read_cached_fields()
        mutations = [(f"missing {key}", key, None) for key in original]
        mutations += [
            ("missing row", "features", original["features"][:-1]),
            ("missing column", "features", original["features"][:, :-1]),
            ("wrong feature dtype", "features", original["features"].astype(np.float64)),
            ("nan feature", "features", np.full((3, 2), np.nan, dtype=np.float32)),
            ("infinite feature", "features", np.full((3, 2), np.inf, dtype=np.float32)),
            ("short gene list", "gene_ids", np.asarray(["gene-a"])),
            ("matrix genes", "gene_ids", np.asarray([["gene-a", "gene-b"]])),
            ("object genes", "gene_ids", np.asarray(["gene-a", "gene-b"], dtype=object)),
        ]
        for label, field, replacement in mutations:
            with self.subTest(mutation=label):
                changed = {key: value.copy() for key, value in original.items()}
                if replacement is None:
                    del changed[field]
                else:
                    changed[field] = replacement
                np.savez_compressed(path, **changed)
                with self.assertRaisesRegex(ValueError, "Invalid expression cache"):
                    dataset_utils._build_expression_cache(self.opt, self.records)

    def test_corrupt_cache_is_reported_instead_of_returning_or_silently_rebuilding(self):
        dataset_utils._build_expression_cache(self.opt, self.records)
        path = self.cache_files()[0]
        path.write_bytes(b"not a valid npz archive")
        with patch.object(dataset_utils, "_read_expression_vector") as read:
            with self.assertRaisesRegex(ValueError, "Invalid expression cache"):
                dataset_utils._build_expression_cache(self.opt, self.records)
            read.assert_not_called()

    def test_old_v2_log_cache_is_ignored_even_when_record_count_matches(self):
        cache_dir = Path(self.opt.data_dir) / "cache"
        cache_dir.mkdir(parents=True)
        np.savez_compressed(
            cache_dir / "tcga_brca_log1p_v2_tpm_unstranded_2_3samples_legacy.npz",
            cache_schema_version=2,
            preprocessing="log1p_only",
            features=np.full((3, 2), -777, dtype=np.float32),
        )
        with patch.object(
            dataset_utils, "_read_expression_vector", wraps=dataset_utils._read_expression_vector
        ) as read:
            actual = dataset_utils._build_expression_cache(self.opt, self.records)
        self.assertEqual(read.call_count, len(self.records))
        np.testing.assert_array_equal(actual[0], np.log1p(self.raw))
        self.assertEqual(len(self.cache_files()), 1)

    def test_different_gene_order_fails_without_publishing_a_cache(self):
        self.write_source(self.records[1]["file_path"], self.raw[1][::-1], genes=("gene-b", "gene-a"))
        with self.assertRaises(ValueError):
            dataset_utils._build_expression_cache(self.opt, self.records)
        self.assertFalse(self.cache_files())

    def test_source_change_during_read_fails_without_publishing_a_cache(self):
        read = dataset_utils._read_expression_vector

        def changing_source(path, column, max_genes):
            result = read(path, column, max_genes)
            with Path(path).open("a") as stream:
                stream.write("\n")
            return result

        with patch.object(dataset_utils, "_read_expression_vector", side_effect=changing_source):
            with self.assertRaises(ValueError):
                dataset_utils._build_expression_cache(self.opt, self.records)
        self.assertFalse(self.cache_files())

    def test_identical_source_bytes_can_be_relocated_without_changing_cache_identity(self):
        expected = dataset_utils._build_expression_cache(self.opt, self.records)
        relocated = copy.deepcopy(self.records)
        for index, record in enumerate(relocated):
            path = self.root / f"relocated-{index}.tsv"
            path.write_bytes(Path(record["file_path"]).read_bytes())
            record["file_path"] = str(path)
        with patch.object(dataset_utils, "_read_expression_vector") as read:
            actual = dataset_utils._build_expression_cache(self.opt, relocated)
        read.assert_not_called()
        self.assertEqual(len(self.cache_files()), 1)
        np.testing.assert_array_equal(actual[0], expected[0])

    def test_failed_atomic_publish_leaves_neither_final_nor_temporary_cache(self):
        with patch.object(dataset_utils.os, "replace", side_effect=OSError("Atomic publication failed")):
            with self.assertRaisesRegex(OSError, "Atomic publication failed"):
                dataset_utils._build_expression_cache(self.opt, self.records)
        self.assertFalse(list((Path(self.opt.data_dir) / "cache").iterdir()))
        actual = dataset_utils._build_expression_cache(self.opt, self.records)
        self.assertEqual(len(self.cache_files()), 1)
        np.testing.assert_array_equal(actual[0], np.log1p(self.raw))

    def test_cached_expression_does_not_hide_a_missing_source_file(self):
        dataset_utils._build_expression_cache(self.opt, self.records)
        Path(self.records[0]["file_path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            dataset_utils._build_expression_cache(self.opt, self.records)


if __name__ == "__main__":
    unittest.main()

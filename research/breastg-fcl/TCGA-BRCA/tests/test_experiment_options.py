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

"""CLI settings must control the experiment or be rejected before preparation."""

import contextlib
import copy
import io
import os
import random
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import job
from configs.TCGA_BRCA import build_parser, parse_args
from prepare_data import prepare_bundles
from utils import dataset_utils


class ExperimentOptionsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())

    def arguments(self, name):
        root = self.root / name
        return [
            "--device",
            "cpu",
            "--output-dir",
            str(root / "output"),
            "--data-dir",
            str(root / "data"),
            "--raw-dir",
            str(root / "raw"),
        ]

    def test_defaults_expose_only_effective_settings(self):
        opt = parse_args(self.arguments("defaults"))
        self.assertEqual(opt.weight_decay, 0.0)
        self.assertEqual(opt.train_split, 0.8)
        self.assertNotIn("test_split", opt)
        self.assertNotIn("use_g_encode", opt)
        help_text = build_parser().format_help()
        self.assertNotIn("--test-split", help_text)
        self.assertNotIn("--use-g-encode", help_text)
        self.assertNotIn("--input-dim", help_text)
        self.assertEqual(opt.input_dim, opt.max_genes)
        for name in ("debug", "load_dir", "log_path", "dataset", "gamma", "gat_rounds", "class_per_task"):
            self.assertNotIn(name, opt)
            self.assertNotIn(f"--{name.replace('_', '-')}", help_text)

    def test_removed_flags_fail_in_config_and_job_before_preparation(self):
        for entrypoint in (parse_args, job.main):
            for flag, value in (
                ("--input-dim", "4"),
                ("--test-split", "0.3"),
                ("--use-g-encode", "false"),
                ("--debug", "true"),
                ("--load-dir", "unused"),
                ("--log-path", "unused"),
                ("--dataset", "unused"),
                ("--gamma", "1"),
                ("--gat-rounds", "1"),
                ("--class-per-task", "1"),
            ):
                with self.subTest(entrypoint=entrypoint.__module__, flag=flag):
                    args = self.arguments("removed") + [flag, value]
                    with (
                        contextlib.redirect_stderr(io.StringIO()) as stderr,
                        patch.object(
                            job,
                            "prepare_bundles",
                            side_effect=AssertionError("Invalid options reached data preparation"),
                        ) as prepare,
                    ):
                        with self.assertRaises(SystemExit) as raised:
                            entrypoint(args)
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn("unrecognized arguments", stderr.getvalue())
                    prepare.assert_not_called()
                    self.assertFalse((self.root / "removed").exists())

    def assert_invalid_before_preparation(self, flag, values):
        for entrypoint in (parse_args, job.main):
            for index, value in enumerate(values):
                with self.subTest(entrypoint=entrypoint.__module__, flag=flag, value=value):
                    name = f"{entrypoint.__module__}-{flag}-{index}"
                    with patch.object(job, "prepare_bundles") as prepare:
                        with self.assertRaisesRegex(ValueError, flag):
                            entrypoint(self.arguments(name) + [f"{flag}={value}"])
                    prepare.assert_not_called()
                    self.assertFalse((self.root / name).exists())

    def test_invalid_training_fractions_fail_before_filesystem_or_preparation(self):
        self.assert_invalid_before_preparation("--train-split", ("0", "1", "-0.2", "1.2", "nan", "inf", "-inf"))

    def test_nonpositive_counts_fail_before_filesystem_or_preparation(self):
        for flag in ("--num-task", "--num-rounds", "--num-local-epochs", "--batch-size"):
            self.assert_invalid_before_preparation(flag, ("0", "-1"))
        for value in ("0", "-1"):
            with self.assertRaisesRegex(ValueError, "--num-clients"):
                parse_args(self.arguments("clients") + ["--num-clients", value])
            with patch.object(job, "prepare_bundles") as prepare, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    job.main(self.arguments("clients") + ["--num-clients", value])
                prepare.assert_not_called()
            self.assertFalse((self.root / "clients").exists())

    def test_dimension_aliases_update_effective_configuration(self):
        for arguments in (["--nc", "3", "--ni", "8"], ["--num-classes", "3", "--nh", "8"]):
            with self.subTest(arguments=arguments):
                opt = parse_args(self.arguments("aliases") + arguments)
                self.assertEqual((opt.num_classes, opt.nc, opt.nh, opt.ni), (3, 3, 8, 8))
                self.assertIn("--nc", build_parser().format_help())
                self.assertIn("--ni", build_parser().format_help())

    def test_alias_and_primary_flags_share_standard_last_value_behavior(self):
        for arguments, expected in (
            (["--nc", "3", "--num-classes", "5", "--ni", "8", "--nh", "16"], (5, 16)),
            (["--num-classes", "5", "--nc", "3", "--nh", "16", "--ni", "8"], (3, 8)),
        ):
            opt = parse_args(self.arguments("alias-order") + arguments)
            self.assertEqual((opt.num_classes, opt.nh), expected)
            self.assertEqual((opt.nc, opt.ni), expected)

    def test_class_alias_rejects_invalid_width_before_preparation(self):
        self.assert_invalid_before_preparation("--nc", ("-1", "0", "1"))

    def test_insufficient_class_width_fails_before_filesystem_or_preparation(self):
        self.assert_invalid_before_preparation("--num-classes", ("-1", "0", "1"))

    def test_invalid_noise_parameters_fail_before_filesystem_or_preparation(self):
        for flag in ("--sensitivity", "--epsilon"):
            self.assert_invalid_before_preparation(flag, ("-1", "nan", "inf", "-inf"))

    def test_overflowing_noise_scale_fails_before_filesystem_or_preparation(self):
        self.assert_invalid_before_preparation("--epsilon", ("1e-320",))

    def test_noise_defaults_and_explicit_zero_modes_are_preserved(self):
        for sensitivity, epsilon, expected in ((1, 1, 1), (2, 4, 0.5), (1, 0, 0), (0, 1, 0)):
            opt = parse_args(self.arguments("noise") + ["--sensitivity", str(sensitivity), "--epsilon", str(epsilon)])
            self.assertEqual(opt.b, expected)

    def test_importing_configuration_does_not_parse_or_create_directories(self):
        with patch("os.makedirs") as makedirs:
            namespace = runpy.run_path(str(Path(PROJECT_DIR) / "configs" / "TCGA_BRCA.py"))
        makedirs.assert_not_called()
        self.assertNotIn("opt", namespace)
        self.assertTrue(callable(namespace["parse_args"]))

    def test_invalid_weight_decay_fails_before_filesystem_or_preparation(self):
        self.assert_invalid_before_preparation("--weight-decay", ("-0.1", "nan", "inf", "-inf"))

    def test_training_fraction_controls_complementary_patient_splits_for_both_strategies(self):
        records, clinical = [], {}
        for index in range(60):
            patient = f"TCGA-ZZ-{index:04d}"
            clinical[patient] = {"ajcc_pathologic_stage": ("Stage I", "Stage II", "Stage III")[index % 3]}
            for label in (0, 1):
                records.append(
                    {
                        "file_id": f"{patient}-{label}",
                        "case_submitter_id": patient,
                        "sample_submitter_id": f"{patient}-{'11A' if label == 0 else '01A'}",
                        "label": label,
                        "sample_type": "Solid Tissue Normal" if label == 0 else "Primary Tumor",
                    }
                )
        targets = np.asarray([record["label"] for record in records])
        all_patients = set(clinical)
        for fraction in (0.5, 0.8):
            for strategy in ("random", "clinical_stage"):
                with self.subTest(fraction=fraction, strategy=strategy):
                    opt = parse_args(
                        self.arguments("partition")
                        + [
                            "--train-split",
                            str(fraction),
                            "--num-clients",
                            "2",
                            "--task-split-strategy",
                            strategy,
                        ]
                    )
                    with patch.object(dataset_utils, "_load_clinical_cases", return_value=clinical):
                        make_tasks = (
                            dataset_utils._make_random_task_indices
                            if strategy == "random"
                            else dataset_utils._make_clinical_stage_task_indices
                        )
                        train, test = make_tasks(copy.deepcopy(records), targets, opt)
                    patients = []
                    for assignments in (train, test):
                        indices = [i for tasks in assignments.values() for members in tasks.values() for i in members]
                        patients.append({records[i]["case_submitter_id"] for i in indices})
                    self.assertEqual(len(patients[0]), round(len(all_patients) * fraction))
                    self.assertEqual(len(patients[1]), len(all_patients) - len(patients[0]))
                    self.assertFalse(patients[0] & patients[1])
                    self.assertEqual(patients[0] | patients[1], all_patients)

    def test_positive_weight_decay_survives_site_and_server_bundle_preparation(self):
        opt = parse_args(
            self.arguments("bundles")
            + [
                "--nc",
                "3",
                "--weight-decay",
                "0.125",
                "--num-clients",
                "2",
                "--num-task",
                "1",
                "--max-genes",
                "4",
                "--ni",
                "8",
                "--noise-dim",
                "3",
                "--batch-size",
                "4",
                "--gat-hidden-dim",
                "8",
                "--gat-embedding-dim",
                "4",
                "--gat-heads",
                "2",
            ]
        )
        workflow, server, sites = prepare_bundles(opt, smoke=True)
        self.assertEqual(server["opt"]["num_classes"], 3)
        self.assertEqual(server["opt"]["nh"], 8)
        self.assertEqual(workflow.clients[0].netF.num_classes, 3)
        self.assertEqual(workflow.clients[0].netG.hidden_dim, 8)
        self.assertEqual(server["opt"]["weight_decay"], 0.125)
        self.assertEqual(workflow.server.optimizer_D.param_groups[0]["weight_decay"], 0.125)
        for site, client in zip(sites, server["clients"]):
            self.assertEqual(site["opt"]["weight_decay"], 0.125)
            self.assertTrue(
                all(g["weight_decay"] == 0.125 for g in client["training_state"]["optimizer"]["param_groups"])
            )


if __name__ == "__main__":
    unittest.main()

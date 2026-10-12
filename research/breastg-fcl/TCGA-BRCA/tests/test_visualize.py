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

"""Training plots must reflect recorded rounds and support normal output paths."""

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import visualize


class VisualizeTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.work_dir = Path(self.temp_dir.name)
        self.rc_context = matplotlib.rc_context()
        self.rc_context.__enter__()
        self.addCleanup(self.rc_context.__exit__, None, None, None)
        self.addCleanup(visualize.plt.close, "all")

    def _write_round_csv(self, task_lengths):
        path = self.work_dir / "round_accuracy.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Round", "Average Accuracy"])
            for task, num_rounds in enumerate(task_lengths):
                for round_index in range(num_rounds):
                    writer.writerow([f"Task {task}, Round {round_index + 1}", f"{60 + task + round_index}%"])
        return path

    def _assert_png(self, path):
        self.assertTrue(path.is_file())
        self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        pixels = visualize.plt.imread(path)
        self.assertGreater(pixels.shape[0], 100)
        self.assertGreater(pixels.shape[1], 100)
        self.assertGreater(float(pixels[..., :3].std()), 0.01)

    def _assert_plot_rounds(self, task_lengths, expected_end_points):
        csv_path = self._write_round_csv(task_lengths)
        output = self.work_dir / "curve.png"
        with mock.patch.object(visualize.plt, "close"):
            visualize.visualize_training_curve(str(csv_path), str(output), figsize=(10, 5), dpi=60)
            axes = visualize.plt.gcf().axes[0]
        labeled_ticks = [
            (int(position), label.get_text())
            for position, label in zip(axes.get_xticks(), axes.get_xticklabels())
            if label.get_text()
        ]
        self.assertEqual(labeled_ticks, [(endpoint, str(endpoint)) for endpoint in expected_end_points])
        self.assertEqual(list(axes.lines[0].get_xdata()), list(range(1, sum(task_lengths) + 1)))
        boundary_positions = [float(line.get_xdata()[0]) for line in axes.lines[1:] if len(line.get_xdata())]
        self.assertEqual(boundary_positions, [endpoint + 0.5 for endpoint in expected_end_points[:-1]])
        self._assert_png(output)

    def test_two_rounds_per_task_labels_actual_cumulative_rounds(self):
        self._assert_plot_rounds([2, 2, 2], [2, 4, 6])

    def test_unequal_task_lengths_labels_actual_cumulative_rounds(self):
        self._assert_plot_rounds([1, 3, 2], [1, 4, 6])

    def test_ten_round_tasks_preserve_existing_labels(self):
        self._assert_plot_rounds([10, 10], [10, 20])

    def _run_cli(self, output_path):
        csv_path = self._write_round_csv([2, 2, 2])
        result = subprocess.run(
            [
                sys.executable,
                str(Path(PROJECT_DIR) / "visualize.py"),
                "--csv_path",
                str(csv_path),
                "--output_path",
                output_path,
                "--dpi",
                "60",
            ],
            cwd=self.work_dir,
            env={**os.environ, "MPLBACKEND": "Agg"},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self._assert_png(self.work_dir / output_path)

    def test_cli_rejects_missing_empty_and_malformed_csv(self):
        path = self.work_dir / "invalid.csv"
        for content in (
            None,
            "",
            "Round,Average Accuracy\n",
            "Round,Average Accuracy\nTask 1,invalid\n",
            "wrong,columns\na,b\n",
            "Round,Average Accuracy\nTask 1,nan\n",
        ):
            with self.subTest(content=content):
                path.unlink(missing_ok=True)
                if content is not None:
                    path.write_text(content)
                output = self.work_dir / "invalid.png"
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(PROJECT_DIR) / "visualize.py"),
                        "--csv_path",
                        str(path),
                        "--output_path",
                        str(output),
                    ],
                    env={**os.environ, "MPLBACKEND": "Agg"},
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse(output.exists())

    def test_cli_saves_to_basename_in_working_directory(self):
        self._run_cli("curve.png")

    def test_cli_creates_nested_output_directory(self):
        self._run_cli("plots/nested/curve.png")


if __name__ == "__main__":
    unittest.main()

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

import pathlib
import shlex
import sys
import unittest
from types import SimpleNamespace

EXAMPLE_DIR = pathlib.Path(__file__).resolve().parents[5] / "examples" / "advanced" / "medgemma"
_EXAMPLE_DIR_STR = str(EXAMPLE_DIR)
_ADDED_EXAMPLE_DIR = False
if _EXAMPLE_DIR_STR not in sys.path:
    sys.path.insert(0, _EXAMPLE_DIR_STR)
    _ADDED_EXAMPLE_DIR = True

from job import _build_train_args  # noqa: E402

if _ADDED_EXAMPLE_DIR:
    sys.path.remove(_EXAMPLE_DIR_STR)


class MedGemmaJobTest(unittest.TestCase):
    def test_build_train_args_quotes_paths_with_spaces(self):
        args = SimpleNamespace(
            model_name_or_path="/models/medgemma 4b",
            max_steps=None,
            num_train_epochs=3,
            learning_rate=2e-4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
        )

        train_args = _build_train_args(
            args=args,
            site_data_path="/tmp/site data/site-1",
            image_root="/tmp/image root",
            report_to="none",
        )

        self.assertEqual(
            shlex.split(train_args),
            [
                "--data_path",
                "/tmp/site data/site-1",
                "--image_root",
                "/tmp/image root",
                "--model_name_or_path",
                "/models/medgemma 4b",
                "--num_train_epochs",
                "3",
                "--learning_rate",
                "0.0002",
                "--per_device_train_batch_size",
                "4",
                "--per_device_eval_batch_size",
                "4",
                "--gradient_accumulation_steps",
                "4",
                "--report_to",
                "none",
            ],
        )

    def test_build_train_args_prefers_max_steps_when_set(self):
        args = SimpleNamespace(
            model_name_or_path="google/medgemma-4b-it",
            max_steps=20,
            num_train_epochs=3,
            learning_rate=2e-4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
        )

        train_args = _build_train_args(
            args=args,
            site_data_path="/tmp/site-1",
            image_root="/tmp/images",
            report_to="wandb",
        )

        parsed = shlex.split(train_args)
        self.assertIn("--max_steps", parsed)
        self.assertNotIn("--num_train_epochs", parsed)


if __name__ == "__main__":
    unittest.main()

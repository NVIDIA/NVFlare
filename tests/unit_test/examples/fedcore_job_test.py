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

import shlex
from pathlib import Path
from types import SimpleNamespace

from tests.unit_test.examples.fedcore_test_utils import fedcore_import_context


def test_fedcore_job_quotes_client_arguments():
    args = SimpleNamespace(
        batch_size=16,
        dropout=0.1,
        effect_weight=1.0,
        hidden_dim=128,
        learning_rate=0.01,
        local_epochs=4,
        seed=7,
        task_weight=1.0,
    )
    cache_dir = Path("/tmp/fedcore cache")
    output_dir = Path("/tmp/fedcore output")

    with fedcore_import_context():
        import job

        train_args = shlex.split(job._build_train_args(args, cache_dir, output_dir, "site-1", 32))

    assert train_args == [
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(output_dir),
        "--site",
        "site-1",
        "--input-dim",
        "32",
        "--hidden-dim",
        "128",
        "--dropout",
        "0.1",
        "--local-epochs",
        "4",
        "--batch-size",
        "16",
        "--learning-rate",
        "0.01",
        "--task-weight",
        "1.0",
        "--effect-weight",
        "1.0",
        "--seed",
        "7",
    ]

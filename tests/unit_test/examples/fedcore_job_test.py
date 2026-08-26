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

import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import torch

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
        num_rounds=2,
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


def test_fedcore_job_exports_client_script_and_src_package_from_another_cwd(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    cache_path = cache_dir / "site-1" / "train.pt"
    cache_path.parent.mkdir(parents=True)
    torch.save(
        {
            "example_ids": ["example-1"],
            "labels": torch.tensor([1]),
            "image_available": torch.tensor([True]),
            "paired_mask": torch.tensor([True]),
            "missing_features": torch.zeros((1, 4)),
            "missing_logits": torch.zeros(1),
            "full_logits": torch.ones(1),
        },
        cache_path,
    )

    with fedcore_import_context():
        import job

        runtime_dir = job._stage_client_runtime(tmp_path / "staging")
        unrelated_dir = tmp_path / "unrelated"
        unrelated_dir.mkdir()
        monkeypatch.chdir(unrelated_dir)
        recipe, input_dim = job.build_recipe(_args(), cache_dir, output_dir, runtime_dir)
        export_dir = tmp_path / "export"
        recipe.export(str(export_dir))

    assert input_dim == 4
    job_root = export_dir / "fedcore-image-completion"
    app_dir = job_root / "app_site-1"
    custom_dir = app_dir / "custom"
    assert (custom_dir / "client.py").is_file()
    assert (custom_dir / "src" / "features.py").is_file()
    assert (custom_dir / "src" / "federated.py").is_file()
    assert not (custom_dir / "src" / "__pycache__").exists()
    client_config = json.loads((app_dir / "config" / "config_fed_client.json").read_text())
    command = client_config["executors"][0]["executor"]["args"]["command"]
    assert command[2] == "custom/client.py"
    assert command[command.index("--cache-dir") + 1] == str(cache_dir)


def _args():
    return SimpleNamespace(
        batch_size=16,
        dropout=0.1,
        effect_weight=1.0,
        hidden_dim=128,
        learning_rate=0.01,
        local_epochs=4,
        num_rounds=2,
        seed=7,
        task_weight=1.0,
    )

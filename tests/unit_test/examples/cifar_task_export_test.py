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

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from nvflare.recipe.spec import Recipe


@pytest.mark.parametrize("algorithm", ["fedavg", "fedopt", "fedprox"])
@pytest.mark.parametrize("lifetime", ["job", "task"])
def test_shipped_cifar_job_bundles_native_checkpoint_modules(tmp_path, monkeypatch, algorithm, lifetime):
    root = Path(__file__).parents[3] / "examples/advanced/cifar10/pt"
    source = root / "src"
    job_file = root / "cifar10-sim" / f"cifar10_{algorithm}" / "job.py"
    for path in (source, job_file.parent):
        monkeypatch.syspath_prepend(str(path))

    def local_module(name):
        return name in {"data", "model", "train_utils", "task_checkpoint"} or name.startswith("data.")

    original_modules = {name: sys.modules.pop(name) for name in list(sys.modules) if local_module(name)}
    try:
        spec = importlib.util.spec_from_file_location(f"cifar_export_{algorithm}", job_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(module, "split_and_save", lambda **kwargs: "/unused-offline-export")
        monkeypatch.setattr(module, "add_experiment_tracking", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "SimEnv", lambda **kwargs: object())
        argv = [str(job_file), "--n_clients", "2", "--num_rounds", "3", "--name", f"cifar-{algorithm}"]
        if lifetime == "task":
            argv.extend(("--execution_lifetime", "task"))
        monkeypatch.setattr(sys, "argv", argv)

        def export(recipe, env):
            recipe.export(str(tmp_path))
            return SimpleNamespace(get_status=lambda: "EXPORTED", get_result=lambda: str(tmp_path))

        monkeypatch.setattr(Recipe, "execute", export)
        module.main()
        config_path = next(tmp_path.rglob("config_fed_client.json"))
        config = json.loads(config_path.read_text())
        assert config["execution_lifetime"] == lifetime
        assert config["format_version"] == (3 if lifetime == "task" else 2)
        custom = config_path.parent.parent / "custom"
        for name in ("task_checkpoint.py", "train_utils.py", "data/cifar10_data_utils.py", "data/stateful_loader.py"):
            assert (custom / name).read_bytes() == (source / name).read_bytes()
    finally:
        for name in list(sys.modules):
            if local_module(name):
                sys.modules.pop(name)
        sys.modules.update(original_modules)

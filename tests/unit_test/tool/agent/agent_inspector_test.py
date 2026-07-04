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
import os

import pytest

from nvflare.tool.agent import inspector as inspector_module
from nvflare.tool.agent.frameworks.lightning import LightningDetector
from nvflare.tool.agent.inspector import (
    InspectState,
    _entry_point_imports_file,
    _evidence_score,
    _FamilyResolver,
    _framework_evidence_tied_to_entry_context,
    _module_names_for_file,
    _resolve_import_from_module,
    inspect_path,
)


def _should_promote_lightning_over_pytorch(state):
    # The PyTorch-family promotion decision now lives in the Lightning detector;
    # exercise it through the same resolver the engine uses.
    return LightningDetector().promote_over_family("pytorch", _FamilyResolver(state))


def test_inspect_static_only_does_not_execute_user_module(tmp_path):
    marker = tmp_path / "import_side_effect"
    script = tmp_path / "train.py"
    script.write_text(
        "import pathlib\n"
        "import torch\n"
        f"pathlib.Path({str(marker)!r}).write_text('executed')\n"
        "\n"
        "def train():\n"
        "    return None\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert not marker.exists()
    assert data["target_type"] == "single_training_script"
    assert data["conversion_state"] == "not_converted"
    assert data["frameworks"][0]["name"] == "pytorch"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_detects_pytorch_lightning_and_recommends_lightning_skill(tmp_path):
    script = tmp_path / "train_lightning.py"
    script.write_text(
        "import torch\n" "import pytorch_lightning as pl\n" "\n" "class Net(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert data["conversion_state"] == "not_converted"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]
    assert data["recommended_next_commands"] == [
        "nvflare agent doctor --format json",
        "Use the nvflare-convert-lightning skill before editing.",
    ]
    assert any(item["kind"] == "lightning_class" for item in data["frameworks"][0]["evidence"])


def test_inspect_detects_lightning_pytorch_trainer_import(tmp_path):
    script = tmp_path / "train_lightning.py"
    script.write_text(
        "from lightning.pytorch import LightningDataModule, Trainer\n"
        "\n"
        "class Data(LightningDataModule):\n"
        "    pass\n"
        "\n"
        "trainer = Trainer(max_epochs=1)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]
    evidence_kinds = {item["kind"] for item in data["frameworks"][0]["evidence"]}
    assert {"import", "lightning_class", "lightning_trainer"} <= evidence_kinds


def test_inspect_detects_top_level_lightning_alias_and_from_import(tmp_path):
    script = tmp_path / "train_lightning.py"
    script.write_text(
        "import lightning as L\n"
        "from lightning import LightningModule, Trainer\n"
        "\n"
        "class AliasNet(L.LightningModule):\n"
        "    pass\n"
        "\n"
        "class ImportedNet(LightningModule):\n"
        "    pass\n"
        "\n"
        "alias_trainer = L.Trainer(max_epochs=1)\n"
        "imported_trainer = Trainer(max_epochs=1)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]
    evidence = data["frameworks"][0]["evidence"]
    assert any(item["kind"] == "lightning_class" and item["value"] == "L.LightningModule" for item in evidence)
    assert any(item["kind"] == "lightning_class" and item["value"] == "LightningModule" for item in evidence)
    assert any(item["kind"] == "lightning_trainer" and item["value"] == "L.Trainer" for item in evidence)
    assert any(item["kind"] == "lightning_trainer" and item["value"] == "Trainer" for item in evidence)


def test_inspect_does_not_treat_relative_local_lightning_import_as_framework(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "lightning.py").write_text(
        "class Trainer:\n" "    pass\n",
        encoding="utf-8",
    )
    (package / "train.py").write_text(
        "from .lightning import Trainer\n" "\n" "def main():\n" "    return Trainer()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["frameworks"] == []
    assert data["skill_selection"]["detected_framework"] is None


def test_inspect_prebinds_module_import_declared_after_function(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "def build_trainer():\n" "    return Trainer(max_epochs=1)\n" "\n" "from lightning.pytorch import Trainer\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    evidence = data["frameworks"][0]["evidence"]
    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert any(item["kind"] == "lightning_trainer" and item["value"] == "Trainer" for item in evidence)


def test_inspect_keeps_function_local_import_bindings_in_their_lexical_scope(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n"
        "class Net(torch.nn.Module):\n"
        "    pass\n"
        "def optional_lightning():\n"
        "    from lightning import Trainer\n"
        "    return Trainer\n"
        "def main(Trainer):\n"
        "    return Net(), Trainer()\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    evidence = {item["name"]: item["evidence"] for item in data["frameworks"]}
    assert data["skill_selection"]["detected_framework"] == "pytorch"
    assert not any(item["kind"] == "lightning_trainer" for item in evidence["pytorch_lightning"])


def test_inspect_treats_absolute_import_shadowed_by_local_module_as_local(tmp_path):
    (tmp_path / "lightning.py").write_text("class Trainer:\n    pass\n", encoding="utf-8")
    (tmp_path / "train.py").write_text(
        "from lightning import Trainer\n" "def main():\n" "    return Trainer()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["frameworks"] == []
    assert data["skill_selection"]["detected_framework"] is None


# Lightning-patch conversion-state cases: each writes one trainer script that
# differs only in the import/patch-call spelling and must be classified as
# client_api_converted. Fields: source, expected_call, exact_calls (assert
# calls == [expected_call] instead of membership), and check_framework
# (assert frameworks[0] is pytorch_lightning).
_LIGHTNING_PATCH_CONVERTED_CASES = [
    pytest.param(
        "import lightning as L\n"
        "import nvflare.client.lightning as flare\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "flare.patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        "flare.patch",
        True,
        True,
        id="classifies_lightning_patched_trainer_as_client_api_converted",
    ),
    pytest.param(
        "import lightning as L\n"
        "from nvflare.client.lightning import patch\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        "patch",
        True,
        True,
        id="classifies_imported_lightning_patch_as_client_api_converted",
    ),
    pytest.param(
        "import lightning as L\n"
        "from nvflare.client.lightning import patch as flare_patch\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "flare_patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        "flare_patch",
        True,
        True,
        id="classifies_aliased_lightning_patch_import_as_client_api_converted",
    ),
    pytest.param(
        "import lightning as L\n"
        "import nvflare.client.lightning as nfl\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "nfl.patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        "nfl.patch",
        False,
        True,
        id="classifies_aliased_lightning_patch_module_as_client_api_converted",
    ),
    pytest.param(
        "import lightning as L\n"
        "import nvflare.client.lightning\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "nvflare.client.lightning.patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        "nvflare.client.lightning.patch",
        False,
        True,
        id="classifies_fully_qualified_lightning_patch_as_client_api_converted",
    ),
    pytest.param(
        "from nemo import lightning as nl\n"
        "import nvflare.client.lightning\n"
        "\n"
        "trainer = nl.Trainer(max_steps=10)\n"
        "nvflare.client.lightning.patch(trainer)\n"
        "trainer.fit(model)\n",
        "nvflare.client.lightning.patch",
        False,
        False,
        id="classifies_fully_qualified_lightning_patch_for_wrapper_trainer_as_converted",
    ),
    pytest.param(
        "import lightning as L\n"
        "from nvflare.client import lightning as flare\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "flare.patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        "flare.patch",
        False,
        True,
        id="classifies_from_import_lightning_module_alias_as_client_api_converted",
    ),
    pytest.param(
        "import lightning as L\n"
        "from nvflare.client import lightning\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "lightning.patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        "lightning.patch",
        False,
        True,
        id="classifies_from_import_lightning_module_as_client_api_converted",
    ),
    # nemo.lightning-style wrapper: the trainer is built via ``nl.Trainer`` which
    # is not a recognized Lightning constructor, but ``flare.patch(trainer)`` is
    # still the definitive conversion signal.
    pytest.param(
        "from nemo import lightning as nl\n"
        "import nvflare.client.lightning as flare\n"
        "\n"
        "trainer = nl.Trainer(max_steps=10)\n"
        "flare.patch(trainer, restore_state=False)\n"
        "trainer.fit(model)\n",
        None,
        False,
        False,
        id="classifies_wrapper_trainer_lightning_patch_as_client_api_converted",
    ),
]


@pytest.mark.parametrize(
    ("source", "expected_call", "exact_calls", "check_framework"), _LIGHTNING_PATCH_CONVERTED_CASES
)
def test_inspect_classifies_lightning_patch_as_client_api_converted(
    tmp_path, source, expected_call, exact_calls, check_framework
):
    script = tmp_path / "client.py"
    script.write_text(source, encoding="utf-8")

    data = inspect_path(script)

    if check_framework:
        assert data["frameworks"][0]["name"] == "pytorch_lightning"
    if exact_calls:
        assert data["flare_integration"]["calls"] == [expected_call]
    elif expected_call is not None:
        assert expected_call in data["flare_integration"]["calls"]
    assert data["conversion_state"] == "client_api_converted"


def test_inspect_does_not_route_unconverted_nemo_wrapper_as_lightning(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "from nemo import lightning as nl\n" "\n" "trainer = nl.Trainer(max_steps=10)\n" "trainer.fit(model)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"] == []
    assert data["conversion_state"] == "unknown"
    assert data["skill_selection"]["recommended_skills"] == []


def test_inspect_keeps_plain_pytorch_routing_separate_from_lightning(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "\n"
        "class Net(torch.nn.Module):\n"
        "    pass\n"
        "\n"
        "loader = DataLoader([])\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert [framework["name"] for framework in data["frameworks"]] == ["pytorch"]
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_mixed_pytorch_workspace_with_incidental_lightning_keeps_pytorch(tmp_path):
    # A plain PyTorch entry point plus incidental Lightning imports should
    # surface the mixed workspace without hiding the PyTorch training script.
    # The active PyTorch class also has more weighted evidence than the three
    # incidental Lightning imports.
    (tmp_path / "train.py").write_text(
        "import torch\n" "\n" "class Net(torch.nn.Module):\n" "    pass\n" "\n" "def main():\n" "    model = Net()\n",
        encoding="utf-8",
    )
    (tmp_path / "optional_utils.py").write_text(
        "import pytorch_lightning\n"
        "import lightning.pytorch\n"
        "from pytorch_lightning.callbacks import ModelCheckpoint\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    framework_by_name = {framework["name"]: framework for framework in data["frameworks"]}
    assert framework_names[0] == "pytorch"
    assert "pytorch_lightning" in framework_names
    assert framework_by_name["pytorch"]["confidence"] > framework_by_name["pytorch_lightning"]["confidence"]
    assert len(framework_by_name["pytorch_lightning"]["evidence"]) > len(framework_by_name["pytorch"]["evidence"])
    assert data["target_type"] == "mixed_framework_workspace"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_incidental_lightning_does_not_demote_ranked_pytorch(tmp_path):
    # When PyTorch already ranks ahead of Lightning, preserving that order keeps
    # unrelated frameworks from becoming the display primary.
    (tmp_path / "train.py").write_text(
        "import torch\n"
        "import torchvision\n"
        "import torchaudio\n"
        "\n"
        "class Net(torch.nn.Module):\n"
        "    pass\n"
        "\n"
        "def train():\n"
        "    return Net()\n",
        encoding="utf-8",
    )
    (tmp_path / "boost_helper.py").write_text(
        "import xgboost\n" "import xgboost as xgb\n",
        encoding="utf-8",
    )
    (tmp_path / "optional_lightning.py").write_text(
        "import pytorch_lightning\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[:3] == ["pytorch", "xgboost", "pytorch_lightning"]
    assert data["target_type"] == "mixed_framework_workspace"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


# PyTorch workspaces with an unrelated or unreachable Lightning helper must
# keep PyTorch routing. Fields: files (relative path -> content) and
# expect_mixed_target (assert target_type == "mixed_framework_workspace",
# only where the original case asserted it).
_KEEPS_PYTORCH_DESPITE_LIGHTNING_HELPER_CASES = [
    # Active PyTorch evidence tied to the entry point keeps unrelated Lightning
    # helpers from taking over routing just because they have more active evidence.
    pytest.param(
        {
            "train.py": (
                "import torch\n"
                "\n"
                "class Net(torch.nn.Module):\n"
                "    pass\n"
                "\n"
                "def train():\n"
                "    return Net()\n"
            ),
            "lit_helper.py": (
                "import pytorch_lightning as pl\n"
                "\n"
                "class Helper(pl.LightningModule):\n"
                "    pass\n"
                "\n"
                "trainer = pl.Trainer(max_epochs=1)\n"
            ),
        },
        True,
        id="unrelated_active_lightning_helper_does_not_outweigh_pytorch_entry_point",
    ),
    pytest.param(
        {
            "train.py": "import torch\n" "\n" "class Net(torch.nn.Module):\n" "    pass\n",
            "lit_helper.py": (
                "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n"
            ),
        },
        True,
        id="sparse_pytorch_entry_with_lightning_helper_class_keeps_pytorch",
    ),
    pytest.param(
        {
            "train.py": "import torch\n" "\n" "def main():\n" "    return None\n",
            "lit_helper.py": (
                "import pytorch_lightning as pl\n"
                "\n"
                "class Helper(pl.LightningModule):\n"
                "    pass\n"
                "\n"
                "trainer = pl.Trainer(max_epochs=1)\n"
            ),
        },
        True,
        id="pytorch_entry_import_with_unrelated_active_lightning_helper_keeps_pytorch",
    ),
    pytest.param(
        {
            "train.py": "def main():\n" "    return None\n",
            "model.py": "import torch\n" "\n" "class Net(torch.nn.Module):\n" "    pass\n",
            "lit_helper.py": (
                "import pytorch_lightning as pl\n"
                "\n"
                "class Helper(pl.LightningModule):\n"
                "    pass\n"
                "\n"
                "trainer = pl.Trainer(max_epochs=1)\n"
            ),
        },
        True,
        id="entry_point_blocks_unreachable_active_lightning_helper_from_fallback",
    ),
    pytest.param(
        {
            "models/train.py": (
                "import lightning.pytorch\n"
                "import torch\n"
                "from torch.utils.data import DataLoader\n"
                "\n"
                "def train():\n"
                "    return DataLoader([])\n"
            ),
            "models/lightning.py": (
                "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n"
            ),
        },
        False,
        id="external_lightning_import_does_not_reach_local_lightning_file",
    ),
    pytest.param(
        {
            "models/train.py": (
                "import lightning.pytorch\n"
                "import torch\n"
                "from torch.utils.data import DataLoader\n"
                "\n"
                "def train():\n"
                "    return DataLoader([])\n"
            ),
            "models/lightning/__init__.py": (
                "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n"
            ),
        },
        False,
        id="external_lightning_import_does_not_reach_local_lightning_package",
    ),
    pytest.param(
        {
            "train.py": (
                "import lightning.pytorch\n"
                "import torch\n"
                "from torch.utils.data import DataLoader\n"
                "\n"
                "def train():\n"
                "    return DataLoader([])\n"
            ),
            "lightning/__init__.py": (
                "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n"
            ),
        },
        False,
        id="external_lightning_import_does_not_reach_top_level_lightning_package",
    ),
    pytest.param(
        {
            "experiment.py": (
                "import torch\n"
                "import torch.nn\n"
                "import torch.optim\n"
                "import torch.utils.data\n"
                "import torchaudio\n"
                "import torchvision\n"
                "\n"
                "DEFAULT_EPOCHS = 1\n"
            ),
            "lightning_helper.py": (
                "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n"
            ),
        },
        False,
        id="unrelated_lightning_helper_does_not_beat_pytorch_import_heavy_workspace",
    ),
]


@pytest.mark.parametrize(("files", "expect_mixed_target"), _KEEPS_PYTORCH_DESPITE_LIGHTNING_HELPER_CASES)
def test_inspect_keeps_pytorch_despite_lightning_helper(tmp_path, files, expect_mixed_target):
    for rel_path, content in files.items():
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch"
    assert "pytorch_lightning" in framework_names
    if expect_mixed_target:
        assert data["target_type"] == "mixed_framework_workspace"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def _lightning_module_and_trainer(alias):
    return (
        f"class Net({alias}.LightningModule):\n"
        "    def configure_optimizers(self):\n"
        "        return None\n"
        "def main():\n"
        f"    {alias}.Trainer(max_epochs=1).fit(Net())\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


# Workspaces where the Lightning code is reachable from (or dominates) the
# entry context, so routing goes to the Lightning skill. Fields: files
# (relative path -> content), target (relative path passed to inspect_path,
# or None for the workspace root), and evidence_file (when set, assert a
# lightning_class evidence item from that file on the primary framework).
_ROUTES_TO_LIGHTNING_CASES = [
    # `from lightning import pytorch as pl` (Lightning 2.x form) alongside torch.
    pytest.param(
        {
            "train.py": "import torch\nimport torch.nn as nn\nfrom lightning import pytorch as pl\n"
            + _lightning_module_and_trainer("pl"),
        },
        None,
        None,
        id="from_lightning_import_pytorch_alias_routes_to_lightning",
    ),
    # `import lightning as L` then `L.pytorch.LightningModule` / `L.pytorch.Trainer`.
    pytest.param(
        {
            "train.py": (
                "import torch\nimport lightning as L\n"
                "class Net(L.pytorch.LightningModule):\n"
                "    def configure_optimizers(self):\n"
                "        return None\n"
                "def main():\n"
                "    L.pytorch.Trainer(max_epochs=1).fit(Net())\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            ),
        },
        None,
        None,
        id="bare_lightning_alias_pytorch_submodule_routes_to_lightning",
    ),
    # PyPA src-layout: entry imports mypkg.loop; the module lives at src/mypkg/loop.py.
    pytest.param(
        {
            "src/mypkg/loop.py": "import lightning.pytorch as pl\n" + _lightning_module_and_trainer("pl"),
            "train.py": (
                "import torch\nimport torch.nn as nn\nfrom mypkg.loop import Net\n"
                "def main():\n    return Net()\nif __name__ == '__main__':\n    main()\n"
            ),
        },
        None,
        None,
        id="src_layout_lightning_reachable_from_entry_routes_to_lightning",
    ),
    # Lightning model in a non-entry module + a torch import (no active torch use)
    # + an unrelated entry point must not default to the PyTorch base.
    pytest.param(
        {
            "litmodel.py": (
                "import torch\nimport lightning.pytorch as pl\n"
                "class LitNet(pl.LightningModule):\n"
                "    def configure_optimizers(self):\n"
                "        return None\n"
            ),
            "run.py": "import json\nif __name__ == '__main__':\n    print(json.dumps({}))\n",
        },
        None,
        None,
        id="dominant_lightning_module_with_unrelated_entry_routes_to_lightning",
    ),
    pytest.param(
        {
            "train.py": "import torch\n" "from model import LitModel\n" "\n" "def main():\n" "    return LitModel()\n",
            "model.py": "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n",
        },
        None,
        None,
        id="split_file_lightning_model_imported_by_entry_point_recommends_lightning",
    ),
    pytest.param(
        {
            "train.py": "import torch\n" "from models import LitModel\n" "\n" "def main():\n" "    return LitModel()\n",
            "models/__init__.py": (
                "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n"
            ),
        },
        None,
        None,
        id="package_lightning_model_imported_by_entry_point_recommends_lightning",
    ),
    pytest.param(
        {
            "train.py": "import torch\n" "from models import LitModel\n" "\n" "def main():\n" "    return LitModel()\n",
            "models/__init__.py": "from .model import LitModel\n",
            "models/model.py": (
                "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n"
            ),
        },
        None,
        None,
        id="package_reexported_lightning_model_with_torch_entry_import_recommends_lightning",
    ),
    pytest.param(
        {
            "train.py": (
                "import torch\n"
                "from torch.utils.data import DataLoader\n"
                "from models import LitModel\n"
                "\n"
                "def main():\n"
                "    loader = DataLoader([])\n"
                "    return LitModel(), loader\n"
            ),
            "models/__init__.py": "from .model import LitModel\n",
            "models/model.py": (
                "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n"
            ),
        },
        None,
        None,
        id="package_reexported_lightning_model_with_active_pytorch_entry_recommends_lightning",
    ),
    pytest.param(
        {
            "models/__init__.py": "",
            "train.py": (
                "import torch\n"
                "from torch.utils.data import DataLoader\n"
                "from models import lightning_model\n"
                "\n"
                "def main():\n"
                "    loader = DataLoader([])\n"
                "    return lightning_model.LitModel(), loader\n"
            ),
            "models/lightning_model.py": (
                "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n"
            ),
        },
        None,
        "models/lightning_model.py",
        id="package_lightning_submodule_imported_by_entry_point_recommends_lightning",
    ),
    pytest.param(
        {
            "train.py": "import torch\n" "from models import *\n" "\n" "def main():\n" "    return LitModel()\n",
            "models/__init__.py": (
                "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n"
            ),
        },
        None,
        None,
        id="package_star_import_can_reach_lightning_model",
    ),
    pytest.param(
        {
            "models/__init__.py": "",
            "models/train.py": (
                "import torch\n"
                "from torch.utils.data import DataLoader\n"
                "from . import model\n"
                "\n"
                "def main():\n"
                "    loader = DataLoader([])\n"
                "    return model.LitModel(), loader\n"
            ),
            "models/model.py": (
                "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n"
            ),
        },
        None,
        "models/model.py",
        id="relative_package_lightning_submodule_imported_by_entry_point_recommends_lightning",
    ),
    # A normal Lightning script imports several torch symbols, so PyTorch import
    # evidence outnumbers Lightning symbols. Lightning still wins.
    pytest.param(
        {
            "train.py": (
                "import torch\n"
                "from torch import nn\n"
                "from torch.utils.data import DataLoader\n"
                "import pytorch_lightning as pl\n"
                "\n"
                "class Net(pl.LightningModule):\n"
                "    pass\n"
                "\n"
                "trainer = pl.Trainer(max_epochs=1)\n"
            ),
        },
        "train.py",
        None,
        id="lightning_script_with_many_torch_imports_recommends_lightning",
    ),
    pytest.param(
        {
            "train.py": (
                "import torch\n"
                "from torch import nn\n"
                "from torch.utils.data import DataLoader\n"
                "import pytorch_lightning as pl\n"
                "\n"
                "class Net(pl.LightningModule):\n"
                "    pass\n"
            ),
        },
        "train.py",
        None,
        id="lightning_module_with_many_torch_imports_recommends_lightning",
    ),
    pytest.param(
        {
            "model.py": (
                "import torch\n"
                "import torch.nn as nn\n"
                "import torch.optim as optim\n"
                "import torchaudio\n"
                "import torchvision\n"
                "from torch import nn\n"
                "from torch.nn import functional as F\n"
                "from torch.optim import Adam\n"
                "from torch.utils.data import DataLoader\n"
                "import pytorch_lightning as pl\n"
                "\n"
                "class LitModel(pl.LightningModule):\n"
                "    pass\n"
            ),
        },
        None,
        None,
        id="lightning_model_file_with_many_torch_imports_recommends_lightning",
    ),
]


@pytest.mark.parametrize(("files", "target", "evidence_file"), _ROUTES_TO_LIGHTNING_CASES)
def test_inspect_routes_to_lightning(tmp_path, files, target, evidence_file):
    for rel_path, content in files.items():
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    data = inspect_path(tmp_path / target if target else tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]
    if evidence_file is not None:
        assert any(
            item["file"] == evidence_file and item["kind"] == "lightning_class"
            for item in data["frameworks"][0]["evidence"]
        )


def test_inspect_cross_family_confidence_tie_prefers_entry_context_framework(tmp_path):
    # sklearn entry point + a torch utility can tie on evidence count; a pure
    # alphabetical tie-break would pick pytorch and recommend the PyTorch skill.
    # The framework whose evidence is tied to the entry point (sklearn) wins, so
    # no conversion skill is recommended for the sklearn-dominant repo.
    (tmp_path / "train.py").write_text(
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.metrics import accuracy_score\n"
        "def main():\n"
        "    LogisticRegression()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (tmp_path / "util.py").write_text(
        "import torch\nfrom torch.utils.data import DataLoader\ndef loader(ds):\n    return DataLoader(ds)\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == []
    assert data["frameworks"][0]["name"] == "sklearn"


def test_inspect_weighted_active_model_outranks_more_incidental_imports(tmp_path):
    (tmp_path / "model.py").write_text(
        "from torch import nn\n" "\n" "class Net(nn.Module):\n" "    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "optional_sklearn.py").write_text(
        "import sklearn\n" "import sklearn.metrics\n" "import sklearn.preprocessing\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    confidence = {item["name"]: item["confidence"] for item in data["frameworks"]}
    assert confidence["pytorch"] > confidence["sklearn"]
    assert data["skill_selection"]["detected_framework"] == "pytorch"


def test_inspect_sorts_frameworks_by_raw_score_before_rounding_confidence(tmp_path):
    state = InspectState(root=tmp_path, redact=True)
    state.framework_evidence = {
        "xgboost": [{"kind": "import"}] * 501,
        "sklearn": [{"kind": "import"}] * 500,
    }

    ranked = inspector_module._rank_frameworks(state)

    assert [item["confidence"] for item in ranked] == [0.7, 0.7]
    assert [item["name"] for item in ranked] == ["xgboost", "sklearn"]


def test_inspect_does_not_use_generic_train_helper_as_the_only_entry_root(tmp_path):
    (tmp_path / "helper.py").write_text(
        "import sklearn\n" "def train():\n" "    return None\n",
        encoding="utf-8",
    )
    (tmp_path / "model.py").write_text(
        "import torch\n" "class Net(torch.nn.Module):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["entry_points"] == []
    assert data["skill_selection"]["detected_framework"] == "pytorch"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_main_guard_requires_equality_and_accepts_symmetric_form(tmp_path):
    (tmp_path / "decoy.py").write_text(
        "import sklearn\n" "if __name__ != '__main__':\n" "    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        "import torch\n" "if '__main__' == __name__:\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert [item["path"] for item in data["entry_points"]] == ["app.py"]
    assert data["entry_points"][0]["main_guard"] is True
    assert data["skill_selection"]["detected_framework"] == "pytorch"


@pytest.mark.parametrize(
    ("entry_name", "entry_source"),
    [
        (
            "main.py",
            "import sklearn\n" "def main():\n" "    return None\n" "if __name__ == '__main__':\n" "    main()\n",
        ),
        ("train.py", "import sklearn\nRUN_TRAINING = True\n"),
    ],
    ids=["main-guard", "conventional-launch-file"],
)
def test_inspect_helper_train_function_does_not_become_preferred_entry(tmp_path, entry_name, entry_source):
    (tmp_path / entry_name).write_text(entry_source, encoding="utf-8")
    (tmp_path / "helper.py").write_text(
        "import torch\n" "\n" "class Net(torch.nn.Module):\n" "    pass\n" "\n" "def train():\n" "    return Net()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["detected_framework"] == "sklearn"
    assert data["skill_selection"]["recommended_skills"] == []


def test_inspect_higher_count_unreachable_torch_helper_does_not_beat_sklearn_entry(tmp_path):
    # Count-based confidence can rank an unreachable torch helper above the
    # sklearn the entry point actually uses. Reachability must win: the torch
    # helper is never imported from the entry point, so the sklearn-dominant repo
    # stays on sklearn and abstains from a (wrong) PyTorch recommendation.
    (tmp_path / "train.py").write_text(
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import train_test_split\n"
        "def main():\n"
        "    LogisticRegression()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    # data.py is never imported by the entry point but has more torch evidence
    # (import + submodule import + call) than sklearn's two imports.
    (tmp_path / "data.py").write_text(
        "import torch\nfrom torch.utils.data import DataLoader\ndef loader(ds):\n    return DataLoader(ds)\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    confidence = {fw["name"]: fw["confidence"] for fw in data["frameworks"]}
    assert confidence["pytorch"] > confidence["sklearn"]  # active torch evidence has the higher score
    assert data["skill_selection"]["detected_framework"] == "sklearn"  # but entry-tied sklearn wins
    assert data["skill_selection"]["recommended_skills"] == []


def test_inspect_stale_src_layout_copy_does_not_steal_entry_reachability(tmp_path):
    # A src-layout copy (src/mypkg/loop.py) shares the stripped module name
    # "mypkg.loop" with an actively imported root-level mypkg/loop.py. The stale
    # copy (Lightning) must not be scored as entry-reachable via the shared name;
    # the entry point imports the root PyTorch module, so routing stays PyTorch.
    (tmp_path / "train.py").write_text(
        "from mypkg.loop import run\nif __name__ == '__main__':\n    run()\n",
        encoding="utf-8",
    )
    (tmp_path / "mypkg").mkdir()
    (tmp_path / "mypkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "mypkg" / "loop.py").write_text(
        "import torch\nimport torch.nn as nn\nclass Net(nn.Module):\n    pass\ndef run():\n    return Net()\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "mypkg").mkdir(parents=True)
    (tmp_path / "src" / "mypkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "mypkg" / "loop.py").write_text(
        "import lightning.pytorch as pl\nclass Lit(pl.LightningModule):\n    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_src_layout_model_imported_by_root_entry_still_resolves(tmp_path):
    # Guard the src-layout fix does not over-correct: with no root-level
    # collision, an entry point that imports mypkg.loop must still reach the
    # src/mypkg/loop.py Lightning model and route to Lightning.
    (tmp_path / "train.py").write_text(
        "from mypkg.loop import Lit\nif __name__ == '__main__':\n    Lit()\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "mypkg").mkdir(parents=True)
    (tmp_path / "src" / "mypkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "mypkg" / "loop.py").write_text(
        "import lightning.pytorch as pl\nclass Lit(pl.LightningModule):\n    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_reachable_lightning_class_wins_over_co_located_torch(tmp_path):
    # DESIGN DECISION: a LightningModule reachable from the entry context routes
    # to the Lightning skill even when co-located with dominant plain-torch code.
    # This deliberately favors the common case (real Lightning projects compose
    # torch models/submodules) over the rare stray-leftover-LightningModule edge,
    # which is low-harm (a Lightning conversion still works). Mis-routing a real
    # Lightning repo to the PyTorch skill would be worse. See the rationale in
    # LightningDetector.promote_over_family. (Previously this asserted PyTorch;
    # the guard that produced that was intentionally removed.)
    (tmp_path / "model.py").write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "import pytorch_lightning as pl\n"
        "class LegacyLit(pl.LightningModule):\n"
        "    pass\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(4, 2)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
        "def train():\n"
        "    net = Net()\n"
        "    opt = torch.optim.SGD(net.parameters(), lr=0.1)\n"
        "    loss = torch.nn.CrossEntropyLoss()\n"
        "    loader = torch.utils.data.DataLoader([])\n"
        "    return net, opt, loss, loader\n"
        "def main():\n"
        "    return train()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_reachable_active_torch_model_beats_import_only_sklearn_entry(tmp_path):
    # #4: when the entry reaches BOTH import-only sklearn and an ACTIVE torch
    # model, prefer the framework with real (active) evidence -> recommend the
    # PyTorch conversion rather than abstaining on the sklearn imports.
    (tmp_path / "train.py").write_text(
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.metrics import accuracy_score\n"
        "from net import Net\n"
        "def main():\n"
        "    LogisticRegression()\n"
        "    Net()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (tmp_path / "net.py").write_text(
        "import torch.nn as nn\nclass Net(nn.Module):\n    def forward(self, x):\n        return x\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["detected_framework"] == "pytorch"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_import_only_sklearn_entry_still_wins_when_torch_unreachable(tmp_path):
    # #4 control (preserves the earlier sklearn-entry decision): when the torch
    # helper is NOT reachable from the entry, the entry-owned sklearn stays
    # primary and no conversion skill is recommended.
    (tmp_path / "train.py").write_text(
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import train_test_split\n"
        "def main():\n"
        "    LogisticRegression()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (tmp_path / "data.py").write_text(
        "import torch\nfrom torch.utils.data import DataLoader\ndef loader(ds):\n    return DataLoader(ds)\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["detected_framework"] == "sklearn"
    assert data["skill_selection"]["recommended_skills"] == []


def test_inspect_frameworks_list_leads_with_detected_primary(tmp_path):
    # #5: frameworks[0] must match detected_framework even when a non-detected
    # framework has higher weighted confidence (here incidental Lightning
    # imports outrank the entry-tied active PyTorch model).
    (tmp_path / "train.py").write_text(
        "import torch\nimport torch.nn as nn\nclass Net(nn.Module):\n    pass\n"
        "def main():\n    Net()\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )
    (tmp_path / "unused.py").write_text(
        "import pytorch_lightning\n"
        "import pytorch_lightning.callbacks\n"
        "import pytorch_lightning.loggers\n"
        "import pytorch_lightning.strategies\n"
        "import pytorch_lightning.utilities\n"
        "import pytorch_lightning.callbacks.progress\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    detected = data["skill_selection"]["detected_framework"]
    confidence = {item["name"]: item["confidence"] for item in data["frameworks"]}
    assert detected == "pytorch"
    assert confidence["pytorch_lightning"] > confidence["pytorch"]
    assert data["frameworks"][0]["name"] == detected


def test_inspect_ranks_on_full_evidence_beyond_display_cap(tmp_path):
    # #3: weighted framework ranking uses all collected evidence, not the display
    # cap of 12. A file with more torch imports than a competing framework's
    # imports ranks PyTorch higher even when both exceed the display cap; the
    # displayed evidence list stays capped.
    torch_imports = "".join(f"import torch.pkg{i}\n" for i in range(20))
    sklearn_imports = "".join(f"import sklearn.pkg{i}\n" for i in range(13))
    (tmp_path / "a.py").write_text(torch_imports, encoding="utf-8")
    (tmp_path / "b.py").write_text(sklearn_imports, encoding="utf-8")

    data = inspect_path(tmp_path)

    confidence = {fw["name"]: fw["confidence"] for fw in data["frameworks"]}
    assert confidence["pytorch"] > confidence["sklearn"]  # 20 vs 13, not a 12-capped tie
    for fw in data["frameworks"]:
        assert len(fw["evidence"]) <= 12  # display still bounded


def test_inspect_incidental_numpy_entry_does_not_suppress_dynamically_loaded_pytorch(tmp_path):
    # An incidental `import numpy` in the entry must not win primary-framework
    # selection over the real PyTorch code, even when that code is loaded
    # dynamically (no static import chain) and lives in a non-entry-point
    # submodule. numpy is a numerical utility, not the training framework.
    (tmp_path / "main.py").write_text(
        "import numpy as np\n"
        "import importlib\n"
        "def main():\n"
        "    importlib.import_module('pkg.net')\n"
        "    return np.array([1])\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "net.py").write_text(
        "import torch\nimport torch.nn as nn\nNET = torch.nn.Linear(4, 2)\nOPT = torch.optim.SGD(NET.parameters(), lr=0.1)\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["detected_framework"] == "pytorch"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_tied_numpy_entry_fallback_prefers_dynamically_loaded_pytorch(tmp_path):
    # A single incidental numpy import and a single dynamically-loaded torch
    # import tie on confidence. The fallback must not route to numpy just
    # because it sorts alphabetically before pytorch.
    (tmp_path / "main.py").write_text(
        "import importlib\n"
        "import numpy as np\n"
        "def main():\n"
        "    importlib.import_module('pkg.net')\n"
        "    return np.array([1])\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "net.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(tmp_path)

    assert {framework["name"]: framework["confidence"] for framework in data["frameworks"]} == {
        "numpy": 0.7,
        "pytorch": 0.7,
    }
    assert data["skill_selection"]["detected_framework"] == "pytorch"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_reverse_src_layout_prefers_importing_files_packaging_root(tmp_path):
    # Reverse of the stale-src-copy case: entry and real code live under src/,
    # and a stale copy sits at the root. The import from src/pkg/main.py must
    # resolve to the src/ copy (sharing its packaging root), not the stale
    # root-level copy, so routing follows the real (src/) PyTorch code.
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "pkg" / "main.py").write_text(
        "from pkg.loop import run\nif __name__ == '__main__':\n    run()\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "pkg" / "loop.py").write_text(
        "import torch\nimport torch.nn as nn\nclass Net(nn.Module):\n    pass\ndef run():\n    return Net()\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "loop.py").write_text(
        "import lightning.pytorch as pl\nclass Lit(pl.LightningModule):\n    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_stray_lightning_import_is_mixed_framework_not_flare_mixed_workspace(tmp_path):
    # A plain PyTorch repo with a stray, unused `import pytorch_lightning` is a
    # mixed-framework workspace, not the FLARE conversion "mixed_workspace".
    (tmp_path / "train.py").write_text(
        "import torch\nimport torch.nn as nn\nclass Net(nn.Module):\n    pass\n"
        "def main():\n    Net()\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )
    (tmp_path / "misc.py").write_text("import pytorch_lightning\n", encoding="utf-8")

    data = inspect_path(tmp_path)

    assert data["target_type"] == "mixed_framework_workspace"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_torch_ops_inside_lightning_module_with_unrelated_entry_routes_to_lightning(tmp_path):
    # A realistic LightningModule calls several torch APIs (optimizer, loss,
    # dataloader). Those live in the same file as the active Lightning evidence,
    # so they are Lightning code, not standalone PyTorch usage. Even though the
    # raw torch-call count exceeds the single LightningModule class, an unrelated
    # entry point must not let that in-Lightning torch usage force the PyTorch
    # base and misroute a genuine Lightning repo.
    (tmp_path / "litmodel.py").write_text(
        "import torch\nimport lightning.pytorch as pl\n"
        "from torch.optim import SGD\nfrom torch.utils.data import DataLoader\n"
        "class LitNet(pl.LightningModule):\n"
        "    def train_dataloader(self):\n"
        "        return DataLoader([])\n"
        "    def training_step(self, batch, batch_idx):\n"
        "        return torch.nn.functional.cross_entropy(batch, batch)\n"
        "    def configure_optimizers(self):\n"
        "        return SGD(self.parameters(), lr=0.1)\n",
        encoding="utf-8",
    )
    (tmp_path / "run.py").write_text(
        "import json\nif __name__ == '__main__':\n    print(json.dumps({}))\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_split_file_pytorch_model_with_unrelated_lightning_helper_keeps_pytorch(tmp_path):
    (tmp_path / "train.py").write_text(
        "from model import Net\n" "\n" "def main():\n" "    return Net()\n",
        encoding="utf-8",
    )
    (tmp_path / "model.py").write_text(
        "import torch\n" "\n" "class Net(torch.nn.Module):\n" "    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "lit_helper.py").write_text(
        "import pytorch_lightning as pl\n"
        "\n"
        "class Helper(pl.LightningModule):\n"
        "    pass\n"
        "\n"
        "trainer = pl.Trainer(max_epochs=1)\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch"
    assert "pytorch_lightning" in framework_names
    assert data["target_type"] == "mixed_framework_workspace"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_top_level_model_import_does_not_reach_nested_model_file(tmp_path):
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    (tmp_path / "train.py").write_text(
        "import torch\n" "import model\n" "\n" "def train():\n" "    return model.Net()\n",
        encoding="utf-8",
    )
    (tmp_path / "model.py").write_text(
        "import torch\n" "\n" "class Net(torch.nn.Module):\n" "    pass\n",
        encoding="utf-8",
    )
    (helpers / "model.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch"
    assert "pytorch_lightning" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_external_lightning_import_does_not_promote_shadowing_lightning_package(tmp_path):
    # An unreachable local ``lightning`` package can carry far more active
    # Lightning evidence than the PyTorch entry point. The dotted external import
    # ``import lightning.pytorch`` must not resolve to that local package, so the
    # entry-context guard -- not the weighted fallback score -- must keep routing
    # on PyTorch even though the helper would otherwise win on raw evidence.
    package = tmp_path / "models"
    package.mkdir()
    (package / "train.py").write_text(
        "import lightning.pytorch\n"
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "\n"
        "def train():\n"
        "    return DataLoader([])\n",
        encoding="utf-8",
    )
    lightning_package = package / "lightning"
    lightning_package.mkdir()
    (lightning_package / "__init__.py").write_text(
        "import pytorch_lightning as pl\n"
        "import lightning.pytorch\n"
        "from pytorch_lightning.callbacks import ModelCheckpoint\n"
        "\n"
        "class HelperA(pl.LightningModule):\n"
        "    pass\n"
        "\n"
        "class HelperB(pl.LightningModule):\n"
        "    pass\n"
        "\n"
        "trainer = pl.Trainer(max_epochs=1)\n"
        "second_trainer = pl.Trainer(max_epochs=2)\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_by_name = {framework["name"]: framework for framework in data["frameworks"]}
    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch"
    assert "pytorch_lightning" in framework_names
    # The helper genuinely carries more raw evidence; routing still stays on
    # PyTorch because the helper is unreachable from the entry point.
    assert len(framework_by_name["pytorch_lightning"]["evidence"]) > len(framework_by_name["pytorch"]["evidence"])
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_same_directory_model_import_can_reach_lightning_helper(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "train.py").write_text(
        "import torch\n" "from model import LitModel\n" "\n" "def main():\n" "    return LitModel()\n",
        encoding="utf-8",
    )
    (package / "model.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_nested_local_dotted_import_can_reach_lightning_helper(tmp_path):
    package = tmp_path / "models"
    package.mkdir()
    (package / "train.py").write_text(
        "import torch\n" "from layers.block import LitModel\n" "\n" "def main():\n" "    return LitModel()\n",
        encoding="utf-8",
    )
    layers = package / "layers"
    layers.mkdir()
    (layers / "__init__.py").write_text("", encoding="utf-8")
    (layers / "block.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_nested_dotted_import_follows_context_resolved_package_init(tmp_path):
    # The Lightning evidence lives in ``models/layers/__init__.py`` and is only reachable through
    # the context-resolved ``models.layers`` package prefix of ``from layers.block import ...`` in
    # ``models/train.py``. An unrelated top-level ``layers/`` package (matching the raw prefix) must
    # not be followed.
    package = tmp_path / "models"
    package.mkdir()
    (package / "train.py").write_text(
        "import torch\n" "from layers.block import LitModel\n" "\n" "def main():\n" "    return LitModel()\n",
        encoding="utf-8",
    )
    nested_layers = package / "layers"
    nested_layers.mkdir()
    (nested_layers / "__init__.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )
    # block.py is neutral so the Lightning evidence in ``models/layers/__init__.py`` is reachable
    # only through the package-prefix follow of the resolved ``models.layers.block`` module.
    (nested_layers / "block.py").write_text("import torch\n", encoding="utf-8")
    unrelated_layers = tmp_path / "layers"
    unrelated_layers.mkdir()
    (unrelated_layers / "__init__.py").write_text("import tensorflow\n", encoding="utf-8")

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_nested_dotted_import_does_not_follow_raw_top_level_package_init(tmp_path):
    # Complement to the test above: the raw top-level prefix ``layers`` now carries the active
    # Lightning evidence, while the context-resolved ``models.layers`` package is plain PyTorch.
    # Following the raw top-level prefix would incorrectly reach the Lightning evidence, so routing
    # must stay on PyTorch to prove only the context-resolved package prefix is traversed.
    package = tmp_path / "models"
    package.mkdir()
    (package / "train.py").write_text(
        "import torch\n" "from layers.block import Model\n" "\n" "def main():\n" "    return Model()\n",
        encoding="utf-8",
    )
    nested_layers = package / "layers"
    nested_layers.mkdir()
    # The context-resolved ``models.layers`` package is plain PyTorch.
    (nested_layers / "__init__.py").write_text(
        "import torch.nn as nn\n" "\n" "class Model(nn.Module):\n" "    pass\n",
        encoding="utf-8",
    )
    (nested_layers / "block.py").write_text("import torch\n", encoding="utf-8")
    # The unrelated top-level ``layers/`` package matches the raw prefix and holds Lightning
    # evidence; it must not be followed from ``models/train.py``.
    unrelated_layers = tmp_path / "layers"
    unrelated_layers.mkdir()
    (unrelated_layers / "__init__.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch"
    # Lightning is still detected globally but stays unreachable, so routing remains PyTorch.
    assert "pytorch_lightning" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_split_file_lightning_trainer_helper_beats_pytorch_entry_point(tmp_path):
    (tmp_path / "train.py").write_text(
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "from lightning_helper import build_trainer\n"
        "\n"
        "def main():\n"
        "    loader = DataLoader([])\n"
        "    return build_trainer(loader)\n",
        encoding="utf-8",
    )
    (tmp_path / "lightning_helper.py").write_text(
        "import pytorch_lightning as pl\n"
        "\n"
        "class LitModel(pl.LightningModule):\n"
        "    pass\n"
        "\n"
        "def build_trainer(_loader):\n"
        "    trainer = pl.Trainer(max_epochs=1)\n"
        "    return trainer, LitModel()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_unqualified_lightning_symbol_without_from_import_stays_import_only(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import pytorch_lightning\n" "\n" "class LitModel(LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert all(item["kind"] == "import" for item in data["frameworks"][0]["evidence"])


def test_inspect_lightning_subscripted_base_recommends_lightning(tmp_path):
    script = tmp_path / "model.py"
    script.write_text(
        "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule[int]):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert any(item["kind"] == "lightning_class" for item in data["frameworks"][0]["evidence"])


def test_module_names_for_file_handles_package_and_invalid_paths():
    assert _module_names_for_file("model.py") == {"model"}
    assert _module_names_for_file("pkg/model.py") == {"pkg.model"}
    assert _module_names_for_file("pkg/__init__.py") == {"pkg"}
    assert _module_names_for_file("notes.txt") == set()
    assert _module_names_for_file("../model.py") == set()


def test_resolve_import_from_module_handles_absolute_and_relative_imports():
    assert _resolve_import_from_module("train.py", "models", 0) == "models"
    assert _resolve_import_from_module("pkg/train.py", "", 1) == "pkg"
    assert _resolve_import_from_module("pkg/train.py", "model", 1) == "pkg.model"
    assert _resolve_import_from_module("pkg/sub/train.py", "model", 2) == "pkg.model"


def test_lightning_routing_helper_defensive_branches(tmp_path):
    state = InspectState(root=tmp_path / "train.py", redact=True)
    state.framework_evidence["pytorch_lightning"] = [
        {"file": "helper.py", "line": 1, "kind": "lightning_class", "value": "pl.LightningModule"}
    ]
    state.framework_evidence["pytorch"] = [
        {"file": "model.py", "line": 1, "kind": "pytorch_class", "value": "torch.nn.Module"}
    ]

    assert not _should_promote_lightning_over_pytorch(state)
    assert not _framework_evidence_tied_to_entry_context(state, state.framework_evidence["pytorch_lightning"])
    assert not _framework_evidence_tied_to_entry_context(state, state.framework_evidence["pytorch"])
    assert not _entry_point_imports_file(state, "README.md")
    assert _evidence_score([{"kind": "unknown"}]) == 1


def test_entry_point_reachability_is_cached_across_evidence_checks(tmp_path, monkeypatch):
    state = InspectState(root=tmp_path, redact=True)
    state.entry_points.append({"path": "main.py", "kind": "python_script", "functions": ["main"], "main_guard": True})
    state.entry_point_priorities["main.py"] = 3
    state.file_imports = {
        "main.py": {"model"},
        "model.py": {"layers"},
        "layers.py": set(),
    }
    calls = 0
    original = inspector_module._local_files_for_import

    def counting_local_files_for_import(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(inspector_module, "_local_files_for_import", counting_local_files_for_import)
    evidence = [{"file": "layers.py", "line": 1, "kind": "pytorch_class", "value": "nn.Module"}]

    for _ in range(100):
        assert _framework_evidence_tied_to_entry_context(state, evidence)

    assert calls == 2
    assert state.reachable_files_cache == {"model.py", "layers.py"}


def test_lightning_routing_fallback_prefers_active_lightning_over_pytorch_imports(tmp_path):
    state = InspectState(root=tmp_path, redact=True)
    state.framework_evidence["pytorch_lightning"] = [
        {"file": "model.py", "line": 1, "kind": "import", "value": "pytorch_lightning"},
        {"file": "model.py", "line": 6, "kind": "lightning_class", "value": "pl.LightningModule"},
    ]
    state.framework_evidence["pytorch"] = [
        {"file": "model.py", "line": 2, "kind": "import", "value": "torch"},
        {"file": "model.py", "line": 3, "kind": "import", "value": "torch.nn"},
        {"file": "model.py", "line": 4, "kind": "import", "value": "torch.optim"},
        {"file": "model.py", "line": 5, "kind": "import", "value": "torch.utils.data"},
    ]

    assert _should_promote_lightning_over_pytorch(state)


def test_lightning_routing_fallback_keeps_pytorch_import_threshold_for_unrelated_helpers(tmp_path):
    state = InspectState(root=tmp_path, redact=True)
    state.framework_evidence["pytorch_lightning"] = [
        {"file": "lightning_helper.py", "line": 1, "kind": "import", "value": "pytorch_lightning"},
        {"file": "lightning_helper.py", "line": 4, "kind": "lightning_class", "value": "pl.LightningModule"},
    ]
    state.framework_evidence["pytorch"] = [
        {"file": "experiment.py", "line": 1, "kind": "import", "value": "torch"},
        {"file": "experiment.py", "line": 2, "kind": "import", "value": "torch.nn"},
        {"file": "experiment.py", "line": 3, "kind": "import", "value": "torch.optim"},
        {"file": "experiment.py", "line": 4, "kind": "import", "value": "torch.utils.data"},
    ]

    assert not _should_promote_lightning_over_pytorch(state)


def test_inspect_unrelated_entry_ignores_pytorch_calls_inside_lightning_module(tmp_path):
    (tmp_path / "main.py").write_text(
        "def main():\n" "    return 'unrelated entry point'\n",
        encoding="utf-8",
    )
    (tmp_path / "helper.py").write_text("import torch\n", encoding="utf-8")
    (tmp_path / "lit.py").write_text(
        "import torch\n"
        "import pytorch_lightning as pl\n"
        "\n"
        "class LitModel(pl.LightningModule):\n"
        "    def configure_optimizers(self):\n"
        "        return torch.optim.SGD(self.parameters(), lr=0.1)\n"
        "\n"
        "    def training_step(self, batch, batch_idx):\n"
        "        loss = torch.nn.CrossEntropyLoss()\n"
        "        return loss\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]
    pytorch_evidence = next(framework["evidence"] for framework in data["frameworks"] if framework["name"] == "pytorch")
    assert any(item["file"] == "lit.py" and item["kind"] == "pytorch_call" for item in pytorch_evidence)


def test_inspect_lightning_fallback_ignores_in_module_torch_calls_for_outside_import(tmp_path):
    # No entry point: active Lightning and standalone PyTorch tie, so the final
    # weighted fallback must not count torch calls from inside the Lightning file.
    (tmp_path / "litmodel.py").write_text(
        "import torch\n"
        "import lightning\n"
        "import pytorch_lightning as pl\n"
        "from pytorch_lightning.callbacks import ModelCheckpoint\n"
        "\n"
        "class LitModel(pl.LightningModule):\n"
        "    def configure_optimizers(self):\n"
        "        return torch.optim.SGD(self.parameters(), lr=0.1)\n"
        "\n"
        "    def train_dataloader(self):\n"
        "        return torch.utils.data.DataLoader([])\n"
        "\n"
        "    def training_step(self, batch, batch_idx):\n"
        "        loss_fn = torch.nn.CrossEntropyLoss()\n"
        "        return loss_fn(batch[0], batch[1])\n",
        encoding="utf-8",
    )
    (tmp_path / "torch_import_only.py").write_text("import torch\n", encoding="utf-8")
    (tmp_path / "base_model.py").write_text(
        "import torch\n" "\n" "class Net(torch.nn.Module):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]
    pytorch_evidence = next(framework["evidence"] for framework in data["frameworks"] if framework["name"] == "pytorch")
    assert any(
        item["file"] == "torch_import_only.py" and item["kind"] == "import" and item["value"] == "torch"
        for item in pytorch_evidence
    )
    assert sum(1 for item in pytorch_evidence if item["file"] == "litmodel.py" and item["kind"] == "pytorch_call") >= 3


@pytest.mark.parametrize(
    ("expected_framework", "training_imports"),
    [
        ("tensorflow", "import tensorflow\nimport keras\nfrom tensorflow.keras import layers\n"),
        ("jax", "import jax\nimport flax\nimport optax\n"),
    ],
)
def test_inspect_non_pytorch_workspace_with_incidental_lightning_import_is_not_lightning(
    tmp_path, expected_framework, training_imports
):
    # The Lightning-over-PyTorch preference is a PyTorch-family rule only. A
    # non-PyTorch workspace with an incidental pytorch_lightning import
    # must not be routed to the Lightning conversion skill.
    (tmp_path / "train.py").write_text(
        training_imports,
        encoding="utf-8",
    )
    (tmp_path / "optional_utils.py").write_text(
        "import pytorch_lightning\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == expected_framework
    assert data["skill_selection"]["detected_framework"] == expected_framework
    assert "nvflare-convert-lightning" not in data["skill_selection"]["recommended_skills"]


def test_inspect_lightning_with_other_frameworks_recommends_lightning(tmp_path):
    # Lightning wins over PyTorch and is surfaced first for display even when a
    # third, higher-import-count framework is present in the workspace.
    (tmp_path / "train.py").write_text(
        "import torch\n"
        "import pytorch_lightning as pl\n"
        "\n"
        "class Net(pl.LightningModule):\n"
        "    pass\n"
        "\n"
        "trainer = pl.Trainer(max_epochs=1)\n",
        encoding="utf-8",
    )
    (tmp_path / "tf_helper.py").write_text(
        "import tensorflow\n" "import keras\n" "from tensorflow.keras import layers\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "tensorflow" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_exported_job_priority_over_lightning_routing(tmp_path):
    app = tmp_path / "app_server"
    app.mkdir()
    (app / "config_fed_server.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "client.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class Net(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert data["conversion_state"] == "exported_job"
    assert data["target_type"] == "exported_submit_ready_flare_job"
    assert data["skill_selection"]["recommended_skills"] == []


def test_inspect_directory_reports_inspected_target_path(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "train.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root)

    assert data["path"] == str(root.resolve(strict=False))
    assert data["path"] != "."


def test_inspect_file_reports_inspected_target_path(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("import torch\n", encoding="utf-8")

    data = inspect_path(script)

    assert data["path"] == str(script.resolve(strict=False))


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_inspect_symlink_reports_link_path_without_resolving_target(tmp_path):
    target_dir = tmp_path / "outside"
    target_dir.mkdir()
    (target_dir / "train.py").write_text("import tensorflow\n", encoding="utf-8")
    link_dir = tmp_path / "linked-repo"
    link_dir.symlink_to(target_dir, target_is_directory=True)

    data = inspect_path(link_dir)

    assert data["path"] == os.path.abspath(os.path.normpath(str(link_dir)))
    assert data["path"] != str(target_dir.resolve(strict=False))
    assert data["scan"]["files_skipped"][0]["code"] == "SYMLINK_SKIPPED"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_inspect_symlinked_file_does_not_classify_target(tmp_path):
    target_file = tmp_path / "outside.py"
    target_file.write_text("import torch\n", encoding="utf-8")
    link_file = tmp_path / "linked-train.py"
    link_file.symlink_to(target_file)

    data = inspect_path(link_file)

    assert data["target_type"] == "unknown_target"
    assert data["frameworks"] == []
    assert data["scan"]["files_skipped"] == [
        {
            "code": "SYMLINK_SKIPPED",
            "path": link_file.name,
            "target": "<REDACTED_PATH>",
            "message": "symlink was not followed during static inspection",
        }
    ]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_inspect_dangling_symlink_is_reported_as_skipped(tmp_path):
    link_file = tmp_path / "dangling-train.py"
    link_file.symlink_to(tmp_path / "missing.py")

    data = inspect_path(link_file)

    assert data["target_type"] == "unknown_target"
    assert data["scan"]["files_skipped"][0]["code"] == "SYMLINK_SKIPPED"
    assert data["scan"]["files_skipped"][0]["path"] == link_file.name


def test_inspect_redacts_secret_literals_and_absolute_paths_by_default(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "API_TOKEN = 'super-secret-value'\n" "DATA_ROOT = '/Users/alice/private/data'\n" "import tensorflow as tf\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    dumped = json.dumps(data)

    assert "super-secret-value" not in dumped
    assert "/Users/alice/private/data" not in dumped
    assert "<REDACTED>" in dumped
    assert "<REDACTED_PATH>" in dumped
    assert data["frameworks"][0]["name"] == "tensorflow"
    assert {finding["code"] for finding in data["findings"]} == {"SECRET_LITERAL_REDACTED"}
    assert data["patterns"]["absolute_data_paths"][0]["code"] == "ABSOLUTE_DATA_PATH"


def test_inspect_redaction_can_be_disabled_for_local_debugging(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("PASSWORD = 'local-debug-secret'\nDATA_ROOT = '/opt/data'\n", encoding="utf-8")

    data = inspect_path(script, redact=False)
    dumped = json.dumps(data)

    assert "local-debug-secret" in dumped
    assert "/opt/data" in dumped


def test_inspect_skips_symlink_without_scanning_target(tmp_path):
    target = tmp_path / "outside.py"
    target.write_text("import tensorflow\n", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "linked.py").symlink_to(target)
    (root / "train.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root)

    assert [framework["name"] for framework in data["frameworks"]] == ["pytorch"]
    assert data["scan"]["files_skipped"][0]["code"] == "SYMLINK_SKIPPED"


def test_inspect_classifies_flare_job_source(tmp_path):
    job_py = tmp_path / "job.py"
    job_py.write_text(
        "from nvflare.recipe import SimEnv\n"
        "\n"
        "def main():\n"
        "    env = SimEnv(num_clients=2)\n"
        "    recipe.export('/tmp/job')\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["target_type"] == "flare_job_source"
    assert data["conversion_state"] == "flare_job"
    assert data["job"]["job_py"] == "job.py"
    assert data["job"]["sim_env_used"] is True
    assert data["job"]["export_support"] is True


def test_inspect_does_not_treat_pytorch_to_call_as_export_support(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n" "\n" "def train(tensor):\n" "    return tensor.to('cpu')\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["job"]["export_support"] is False
    assert "python job.py --export --export-dir <job-dir>" not in data["recommended_next_commands"]


def test_inspect_does_not_treat_builtin_compile_as_torch_compile(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n" "\n" "def build_code():\n" "    return compile('x = 1', '<inline>', 'exec')\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch"
    assert not any(item["kind"] == "torch_compile" for item in data["patterns"]["dynamic"])


def test_inspect_stops_and_caps_skips_after_file_limit(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(20):
        (root / f"train_{index:02d}.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root, max_files=3)

    assert data["scan"]["entries_visited"] == 3
    assert data["scan"]["files_considered"] == 3
    assert data["scan"]["files_scanned"] == 3
    assert data["scan"]["files_skipped_count"] == 1
    assert data["scan"]["files_skipped_truncated"] is False
    assert data["scan"]["files_skipped"] == [
        {"code": "FILE_LIMIT_REACHED", "path": "train_03.py", "message": "file scan limit reached"}
    ]


def test_inspect_file_limit_records_unvisited_stack_directories(tmp_path):
    root = tmp_path / "repo"
    nested = root / "a_nested"
    nested.mkdir(parents=True)
    (nested / "train_nested.py").write_text("import torch\n", encoding="utf-8")
    for index in range(5):
        (root / f"train_{index:02d}.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root, max_files=3)

    skipped = {(entry["code"], entry["path"]) for entry in data["scan"]["files_skipped"]}
    assert ("FILE_LIMIT_REACHED", "train_03.py") in skipped
    assert ("DIRECTORY_NOT_SCANNED_FILE_LIMIT", "a_nested") in skipped
    assert data["scan"]["files_skipped_count"] == 2


def test_inspect_file_limit_records_pending_directories_when_last_child_reaches_limit(tmp_path):
    root = tmp_path / "repo"
    nested = root / "a_nested"
    nested.mkdir(parents=True)
    (nested / "train_nested.py").write_text("import torch\n", encoding="utf-8")
    for index in range(3):
        (root / f"train_{index:02d}.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root, max_files=3)

    skipped = {(entry["code"], entry["path"]) for entry in data["scan"]["files_skipped"]}
    assert ("DIRECTORY_NOT_SCANNED_FILE_LIMIT", "a_nested") in skipped
    assert all(code != "FILE_LIMIT_REACHED" for code, _path in skipped)
    assert data["scan"]["entries_visited"] == 3
    assert data["scan"]["files_scanned"] == 3
    assert data["scan"]["files_skipped_count"] == 1


def test_inspect_file_limit_counts_non_python_entries(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(5):
        (root / f"metadata_{index:02d}.json").write_text("{}\n", encoding="utf-8")
    (root / "train.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root, max_files=3)

    assert data["scan"]["entries_visited"] == 3
    assert data["scan"]["files_considered"] == 3
    assert data["scan"]["files_scanned"] == 0
    assert data["scan"]["files_skipped"] == [
        {"code": "FILE_LIMIT_REACHED", "path": "metadata_03.json", "message": "file scan limit reached"}
    ]


def test_inspect_bounds_directory_count(tmp_path, monkeypatch):
    monkeypatch.setattr(inspector_module, "MAX_DIRECTORIES_VISITED", 1)
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "train.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(tmp_path)

    assert data["scan"]["files_scanned"] == 0
    assert data["scan"]["directories_discovered"] == 1
    assert any(item["code"] == "DIRECTORY_LIMIT_REACHED" for item in data["scan"]["files_skipped"])


def test_inspect_bounds_entries_per_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(inspector_module, "MAX_DIRECTORY_ENTRIES", 2)
    for index in range(3):
        (tmp_path / f"module_{index}.py").write_text("", encoding="utf-8")

    data = inspect_path(tmp_path)

    assert data["scan"]["entries_visited"] == 0
    assert data["completeness"]["complete"] is False
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]
    assert any(item["code"] == "DIRECTORY_ENTRY_LIMIT_REACHED" for item in data["scan"]["files_skipped"])


def test_inspect_directory_entry_cap_is_deterministic_across_creation_order(tmp_path, monkeypatch):
    monkeypatch.setattr(inspector_module, "MAX_DIRECTORY_ENTRIES", 2)
    contents = {"a.py": "", "m.py": "", "z.py": "import torch\n"}
    roots = [tmp_path / "forward", tmp_path / "reverse"]
    orders = [list(contents), list(reversed(contents))]
    results = []
    for root, order in zip(roots, orders):
        root.mkdir()
        for name in order:
            root.joinpath(name).write_text(contents[name], encoding="utf-8")
        results.append(inspect_path(root))

    assert [result["frameworks"] for result in results] == [[], []]
    assert results[0]["scan"] == results[1]["scan"]
    assert results[0]["completeness"] == results[1]["completeness"]
    assert results[0]["skill_selection"] == results[1]["skill_selection"]


def test_inspect_bounds_directory_depth(tmp_path, monkeypatch):
    monkeypatch.setattr(inspector_module, "MAX_DIRECTORY_DEPTH", 1)
    nested = tmp_path / "level_one" / "level_two"
    nested.mkdir(parents=True)
    (nested / "train.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(tmp_path)

    assert data["scan"]["files_scanned"] == 0
    assert any(item["code"] == "DIRECTORY_DEPTH_LIMIT_REACHED" for item in data["scan"]["files_skipped"])


def test_inspect_bounds_ast_nodes(tmp_path, monkeypatch):
    monkeypatch.setattr(inspector_module, "MAX_AST_NODES", 4)
    script = tmp_path / "train.py"
    script.write_text("import torch\ndef train():\n    return torch.tensor(1)\n", encoding="utf-8")

    data = inspect_path(script)

    assert data["frameworks"] == []
    assert any(item["code"] == "AST_NODE_LIMIT_REACHED" for item in data["scan"]["files_skipped"])


def test_inspect_reports_framework_evidence_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(inspector_module, "MAX_EVIDENCE_COLLECT", 2)
    script = tmp_path / "model.py"
    script.write_text("import torch\nimport torch.nn\nimport torch.optim\n", encoding="utf-8")

    data = inspect_path(script)

    assert len(data["frameworks"][0]["evidence"]) == 2
    assert any(item["code"] == "FRAMEWORK_EVIDENCE_LIMIT_REACHED" for item in data["findings"])
    assert data["completeness"]["complete"] is False
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is not available on this platform")
def test_inspect_rejects_python_fifo_without_blocking(tmp_path):
    fifo = tmp_path / "train.py"
    os.mkfifo(fifo)

    data = inspect_path(tmp_path)

    assert data["scan"]["files_scanned"] == 0
    assert data["completeness"]["complete"] is False
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]
    assert any(item["code"] == "NON_REGULAR_FILE" for item in data["scan"]["files_skipped"])


def test_inspect_opens_python_file_once_with_no_follow_and_nonblocking_flags(tmp_path, monkeypatch):
    script = tmp_path / "train.py"
    script.write_text("import torch\n", encoding="utf-8")
    real_open = inspector_module.os.open
    calls = []

    def recording_open(path, flags, *args, **kwargs):
        calls.append((path, flags))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(inspector_module.os, "open", recording_open)

    data = inspect_path(script)

    assert data["scan"]["files_scanned"] == 1
    assert len(calls) == 1
    assert calls[0][1] & getattr(os, "O_NONBLOCK", 0) == getattr(os, "O_NONBLOCK", 0)
    assert calls[0][1] & getattr(os, "O_NOFOLLOW", 0) == getattr(os, "O_NOFOLLOW", 0)


def test_inspect_caps_findings_and_evidence_with_explicit_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(inspector_module, "MAX_EVIDENCE_COLLECT", 2)
    script = tmp_path / "main.py"
    script.write_text(
        "api_key_one = '/private/one'\n" "api_key_two = '/private/two'\n" "api_key_three = '/private/three'\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    completeness = data["completeness"]
    assert completeness["complete"] is False
    assert {"findings", "absolute_path_findings"} <= set(completeness["truncated_buckets"])
    assert completeness["bucket_counts"]["findings"] == {"observed": 4, "collected": 2}
    assert completeness["bucket_counts"]["absolute_path_findings"] == {"observed": 3, "collected": 2}
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_incomplete_import_graph_suppresses_conversion_recommendation(tmp_path, monkeypatch):
    monkeypatch.setattr(inspector_module, "MAX_IMPORTS_PER_FILE", 1)
    (tmp_path / "train.py").write_text(
        "import torch\n" "import local_model\n",
        encoding="utf-8",
    )
    (tmp_path / "local_model.py").write_text("", encoding="utf-8")

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["detected_framework"] == "pytorch"
    assert data["completeness"]["complete"] is False
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]
    assert data["recommended_next_commands"] == ["nvflare agent doctor --format json"]


def test_reachability_budget_counts_only_resolved_local_edges(tmp_path, monkeypatch):
    monkeypatch.setattr(inspector_module, "MAX_REACHABILITY_EDGES", 1)
    external_imports = "".join(f"import external_{index}\n" for index in range(20))
    (tmp_path / "main.py").write_text(external_imports + "from model import Net\n", encoding="utf-8")
    (tmp_path / "model.py").write_text(
        "import torch\n" "class Net(torch.nn.Module):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["completeness"]["complete"] is True
    assert data["skill_selection"]["detected_framework"] == "pytorch"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]

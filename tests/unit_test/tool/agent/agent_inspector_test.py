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


def test_inspect_classifies_lightning_patched_trainer_as_client_api_converted(tmp_path):
    script = tmp_path / "client.py"
    script.write_text(
        "import lightning as L\n"
        "import nvflare.client.lightning as flare\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "flare.patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert data["flare_integration"]["calls"] == ["flare.patch"]
    assert data["conversion_state"] == "client_api_converted"


def test_inspect_classifies_imported_lightning_patch_as_client_api_converted(tmp_path):
    script = tmp_path / "client.py"
    script.write_text(
        "import lightning as L\n"
        "from nvflare.client.lightning import patch\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert data["flare_integration"]["calls"] == ["patch"]
    assert data["conversion_state"] == "client_api_converted"


def test_inspect_classifies_aliased_lightning_patch_import_as_client_api_converted(tmp_path):
    script = tmp_path / "client.py"
    script.write_text(
        "import lightning as L\n"
        "from nvflare.client.lightning import patch as flare_patch\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "flare_patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert data["flare_integration"]["calls"] == ["flare_patch"]
    assert data["conversion_state"] == "client_api_converted"


def test_inspect_classifies_aliased_lightning_patch_module_as_client_api_converted(tmp_path):
    script = tmp_path / "client.py"
    script.write_text(
        "import lightning as L\n"
        "import nvflare.client.lightning as nfl\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "nfl.patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert "nfl.patch" in data["flare_integration"]["calls"]
    assert data["conversion_state"] == "client_api_converted"


def test_inspect_classifies_fully_qualified_lightning_patch_as_client_api_converted(tmp_path):
    script = tmp_path / "client.py"
    script.write_text(
        "import lightning as L\n"
        "import nvflare.client.lightning\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "nvflare.client.lightning.patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert "nvflare.client.lightning.patch" in data["flare_integration"]["calls"]
    assert data["conversion_state"] == "client_api_converted"


def test_inspect_classifies_fully_qualified_lightning_patch_for_wrapper_trainer_as_converted(tmp_path):
    script = tmp_path / "client.py"
    script.write_text(
        "from nemo import lightning as nl\n"
        "import nvflare.client.lightning\n"
        "\n"
        "trainer = nl.Trainer(max_steps=10)\n"
        "nvflare.client.lightning.patch(trainer)\n"
        "trainer.fit(model)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert "nvflare.client.lightning.patch" in data["flare_integration"]["calls"]
    assert data["conversion_state"] == "client_api_converted"


def test_inspect_classifies_from_import_lightning_module_alias_as_client_api_converted(tmp_path):
    script = tmp_path / "client.py"
    script.write_text(
        "import lightning as L\n"
        "from nvflare.client import lightning as flare\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "flare.patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert "flare.patch" in data["flare_integration"]["calls"]
    assert data["conversion_state"] == "client_api_converted"


def test_inspect_classifies_from_import_lightning_module_as_client_api_converted(tmp_path):
    script = tmp_path / "client.py"
    script.write_text(
        "import lightning as L\n"
        "from nvflare.client import lightning\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "lightning.patch(trainer)\n"
        "trainer.fit(model, datamodule=data)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch_lightning"
    assert "lightning.patch" in data["flare_integration"]["calls"]
    assert data["conversion_state"] == "client_api_converted"


def test_inspect_classifies_wrapper_trainer_lightning_patch_as_client_api_converted(tmp_path):
    # nemo.lightning-style wrapper: the trainer is built via ``nl.Trainer`` which
    # is not a recognized Lightning constructor, but ``flare.patch(trainer)`` is
    # still the definitive conversion signal.
    script = tmp_path / "client.py"
    script.write_text(
        "from nemo import lightning as nl\n"
        "import nvflare.client.lightning as flare\n"
        "\n"
        "trainer = nl.Trainer(max_steps=10)\n"
        "flare.patch(trainer, restore_state=False)\n"
        "trainer.fit(model)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

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
    # surface the mixed workspace without hiding the PyTorch training script,
    # even when the helper has more raw Lightning import evidence.
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
    assert framework_by_name["pytorch_lightning"]["confidence"] > framework_by_name["pytorch"]["confidence"]
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


def test_inspect_unrelated_active_lightning_helper_does_not_outweigh_pytorch_entry_point(tmp_path):
    # Active PyTorch evidence tied to the entry point keeps unrelated Lightning
    # helpers from taking over routing just because they have more active evidence.
    (tmp_path / "train.py").write_text(
        "import torch\n" "\n" "class Net(torch.nn.Module):\n" "    pass\n" "\n" "def train():\n" "    return Net()\n",
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
    assert "pytorch" in framework_names
    assert data["target_type"] == "mixed_framework_workspace"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_sparse_pytorch_entry_with_lightning_helper_class_keeps_pytorch(tmp_path):
    (tmp_path / "train.py").write_text(
        "import torch\n" "\n" "class Net(torch.nn.Module):\n" "    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "lit_helper.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch"
    assert "pytorch_lightning" in framework_names
    assert data["target_type"] == "mixed_framework_workspace"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_pytorch_entry_import_with_unrelated_active_lightning_helper_keeps_pytorch(tmp_path):
    (tmp_path / "train.py").write_text(
        "import torch\n" "\n" "def main():\n" "    return None\n",
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


def test_inspect_entry_point_blocks_unreachable_active_lightning_helper_from_fallback(tmp_path):
    (tmp_path / "train.py").write_text(
        "def main():\n" "    return None\n",
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


def test_inspect_from_lightning_import_pytorch_alias_routes_to_lightning(tmp_path):
    # `from lightning import pytorch as pl` (Lightning 2.x form) alongside torch.
    (tmp_path / "train.py").write_text(
        "import torch\nimport torch.nn as nn\nfrom lightning import pytorch as pl\n"
        + _lightning_module_and_trainer("pl"),
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_bare_lightning_alias_pytorch_submodule_routes_to_lightning(tmp_path):
    # `import lightning as L` then `L.pytorch.LightningModule` / `L.pytorch.Trainer`.
    (tmp_path / "train.py").write_text(
        "import torch\nimport lightning as L\n"
        "class Net(L.pytorch.LightningModule):\n"
        "    def configure_optimizers(self):\n"
        "        return None\n"
        "def main():\n"
        "    L.pytorch.Trainer(max_epochs=1).fit(Net())\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_src_layout_lightning_reachable_from_entry_routes_to_lightning(tmp_path):
    # PyPA src-layout: entry imports mypkg.loop; the module lives at src/mypkg/loop.py.
    (tmp_path / "src" / "mypkg").mkdir(parents=True)
    (tmp_path / "src" / "mypkg" / "loop.py").write_text(
        "import lightning.pytorch as pl\n" + _lightning_module_and_trainer("pl"),
        encoding="utf-8",
    )
    (tmp_path / "train.py").write_text(
        "import torch\nimport torch.nn as nn\nfrom mypkg.loop import Net\n"
        "def main():\n    return Net()\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_dominant_lightning_module_with_unrelated_entry_routes_to_lightning(tmp_path):
    # Lightning model in a non-entry module + a torch import (no active torch use)
    # + an unrelated entry point must not default to the PyTorch base.
    (tmp_path / "litmodel.py").write_text(
        "import torch\nimport lightning.pytorch as pl\n"
        "class LitNet(pl.LightningModule):\n"
        "    def configure_optimizers(self):\n"
        "        return None\n",
        encoding="utf-8",
    )
    (tmp_path / "run.py").write_text(
        "import json\nif __name__ == '__main__':\n    print(json.dumps({}))\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


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

    assert data["frameworks"][0]["name"] == "pytorch"  # higher raw count
    assert data["skill_selection"]["detected_framework"] == "sklearn"  # but entry-tied wins
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


def test_inspect_split_file_lightning_model_imported_by_entry_point_recommends_lightning(tmp_path):
    (tmp_path / "train.py").write_text(
        "import torch\n" "from model import LitModel\n" "\n" "def main():\n" "    return LitModel()\n",
        encoding="utf-8",
    )
    (tmp_path / "model.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
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


def test_inspect_external_lightning_import_does_not_reach_local_lightning_file(tmp_path):
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
    (package / "lightning.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch"
    assert "pytorch_lightning" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_external_lightning_import_does_not_reach_local_lightning_package(tmp_path):
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


def test_inspect_external_lightning_import_does_not_reach_top_level_lightning_package(tmp_path):
    (tmp_path / "train.py").write_text(
        "import lightning.pytorch\n"
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "\n"
        "def train():\n"
        "    return DataLoader([])\n",
        encoding="utf-8",
    )
    lightning_package = tmp_path / "lightning"
    lightning_package.mkdir()
    (lightning_package / "__init__.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch"
    assert "pytorch_lightning" in framework_names
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


def test_inspect_package_lightning_model_imported_by_entry_point_recommends_lightning(tmp_path):
    package = tmp_path / "models"
    package.mkdir()
    (tmp_path / "train.py").write_text(
        "import torch\n" "from models import LitModel\n" "\n" "def main():\n" "    return LitModel()\n",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_package_reexported_lightning_model_with_torch_entry_import_recommends_lightning(tmp_path):
    package = tmp_path / "models"
    package.mkdir()
    (tmp_path / "train.py").write_text(
        "import torch\n" "from models import LitModel\n" "\n" "def main():\n" "    return LitModel()\n",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text("from .model import LitModel\n", encoding="utf-8")
    (package / "model.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_package_reexported_lightning_model_with_active_pytorch_entry_recommends_lightning(tmp_path):
    package = tmp_path / "models"
    package.mkdir()
    (tmp_path / "train.py").write_text(
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "from models import LitModel\n"
        "\n"
        "def main():\n"
        "    loader = DataLoader([])\n"
        "    return LitModel(), loader\n",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text("from .model import LitModel\n", encoding="utf-8")
    (package / "model.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_package_lightning_submodule_imported_by_entry_point_recommends_lightning(tmp_path):
    package = tmp_path / "models"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "train.py").write_text(
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "from models import lightning_model\n"
        "\n"
        "def main():\n"
        "    loader = DataLoader([])\n"
        "    return lightning_model.LitModel(), loader\n",
        encoding="utf-8",
    )
    (package / "lightning_model.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]
    lightning_evidence = data["frameworks"][0]["evidence"]
    assert any(
        item["file"] == "models/lightning_model.py" and item["kind"] == "lightning_class" for item in lightning_evidence
    )


def test_inspect_package_star_import_can_reach_lightning_model(tmp_path):
    package = tmp_path / "models"
    package.mkdir()
    (tmp_path / "train.py").write_text(
        "import torch\n" "from models import *\n" "\n" "def main():\n" "    return LitModel()\n",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class LitModel(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_relative_package_lightning_submodule_imported_by_entry_point_recommends_lightning(tmp_path):
    package = tmp_path / "models"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "train.py").write_text(
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "from . import model\n"
        "\n"
        "def main():\n"
        "    loader = DataLoader([])\n"
        "    return model.LitModel(), loader\n",
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
    lightning_evidence = data["frameworks"][0]["evidence"]
    assert any(item["file"] == "models/model.py" and item["kind"] == "lightning_class" for item in lightning_evidence)


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


def test_inspect_lightning_script_with_many_torch_imports_recommends_lightning(tmp_path):
    # A normal Lightning script imports several torch symbols, so PyTorch import
    # evidence outnumbers Lightning symbols. Lightning still wins.
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n"
        "from torch import nn\n"
        "from torch.utils.data import DataLoader\n"
        "import pytorch_lightning as pl\n"
        "\n"
        "class Net(pl.LightningModule):\n"
        "    pass\n"
        "\n"
        "trainer = pl.Trainer(max_epochs=1)\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_lightning_module_with_many_torch_imports_recommends_lightning(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n"
        "from torch import nn\n"
        "from torch.utils.data import DataLoader\n"
        "import pytorch_lightning as pl\n"
        "\n"
        "class Net(pl.LightningModule):\n"
        "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_lightning_model_file_with_many_torch_imports_recommends_lightning(tmp_path):
    script = tmp_path / "model.py"
    script.write_text(
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
        "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch_lightning"
    assert "pytorch" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_inspect_unrelated_lightning_helper_does_not_beat_pytorch_import_heavy_workspace(tmp_path):
    (tmp_path / "experiment.py").write_text(
        "import torch\n"
        "import torch.nn\n"
        "import torch.optim\n"
        "import torch.utils.data\n"
        "import torchaudio\n"
        "import torchvision\n"
        "\n"
        "DEFAULT_EPOCHS = 1\n",
        encoding="utf-8",
    )
    (tmp_path / "lightning_helper.py").write_text(
        "import pytorch_lightning as pl\n" "\n" "class Helper(pl.LightningModule):\n" "    pass\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    framework_names = [framework["name"] for framework in data["frameworks"]]
    assert framework_names[0] == "pytorch"
    assert "pytorch_lightning" in framework_names
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


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

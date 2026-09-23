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
from pathlib import Path

import pytest


def _load_module():
    module_path = Path(__file__).parents[1] / "prepare_base_checkpoint.py"
    spec = importlib.util.spec_from_file_location("evo2_prepare_base_checkpoint", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_checkpoint(module, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("1\n", encoding="utf-8")
    (checkpoint / "latest_train_state.pt").write_bytes(b"latest train state")
    iteration = checkpoint / "iter_0000001"
    iteration.mkdir()
    (iteration / ".metadata").write_bytes(b"distributed metadata")
    (iteration / "__0_0.distcp").write_bytes(b"distributed shard")
    (iteration / "common.pt").write_bytes(b"common state")
    (iteration / "train_state.pt").write_bytes(b"train state")
    (iteration / "metadata.json").write_text(json.dumps(module.EXPECTED_CHECKPOINT_METADATA), encoding="utf-8")
    (iteration / "run_config.yaml").write_text(
        """checkpoint:
  ckpt_format: torch_dist
dataset:
  seq_length: 4096
model:
  _target_: bionemo.evo2.models.evo2_provider.Hyena1bModelProvider
  bf16: true
  ffn_hidden_size: 5120
  hidden_size: 1920
  num_attention_heads: 15
  num_layers: 25
  seq_length: 8192
  tokenizer_library: byte-level
  vocab_size: 512
""",
        encoding="utf-8",
    )
    tokenizer_dir = iteration / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_config = dict(module.EXPECTED_TOKENIZER_CONFIG)
    (tokenizer_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    vocabulary = dict(module.EXPECTED_SPECIAL_TOKEN_IDS)
    vocabulary.update({f"TOKEN_{token_id}": token_id for token_id in range(5, 512)})
    tokenizer = {"version": "1.0", "model": {"type": "WordLevel", "vocab": vocabulary}}
    (tokenizer_dir / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    return checkpoint


def _complete_checkpoint(module, tmp_path):
    checkpoint = _write_checkpoint(module, tmp_path)
    module._write_conversion_provenance(checkpoint)
    return checkpoint


def test_validate_checkpoint_requires_iteration_marker(tmp_path):
    module = _load_module()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    with pytest.raises(ValueError, match="missing"):
        module.validate_checkpoint(checkpoint)

    (checkpoint / "latest_checkpointed_iteration.txt").write_text("not-an-integer", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid iteration"):
        module.validate_checkpoint(checkpoint)

    (checkpoint / "latest_checkpointed_iteration.txt").write_text("2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="iteration must be 1, observed 2"):
        module.validate_checkpoint(checkpoint)

    (checkpoint / "latest_checkpointed_iteration.txt").write_text("1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="iteration is missing"):
        module.validate_checkpoint(checkpoint)


def test_validate_checkpoint_accepts_complete_pinned_conversion(tmp_path):
    module = _load_module()
    checkpoint = _complete_checkpoint(module, tmp_path)

    assert module.validate_checkpoint(checkpoint) == checkpoint.resolve()
    provenance = json.loads((checkpoint / module.PROVENANCE_FILE).read_text(encoding="utf-8"))
    assert provenance["source_model_tag"] == module.MODEL_TAG
    assert provenance["conversion"] == {
        "mixed_precision_recipe": "bf16_mixed",
        "model_size": "evo2_1b_base",
        "sequence_length": 8192,
    }
    assert provenance["tokenizer"]["special_token_ids"] == module.EXPECTED_SPECIAL_TOKEN_IDS
    assert {entry["path"] for entry in provenance["inventory"]} == {
        "iter_0000001/.metadata",
        "iter_0000001/__0_0.distcp",
        "iter_0000001/common.pt",
        "iter_0000001/metadata.json",
        "iter_0000001/run_config.yaml",
        "iter_0000001/tokenizer/tokenizer.json",
        "iter_0000001/tokenizer/tokenizer_config.json",
        "iter_0000001/train_state.pt",
        "latest_checkpointed_iteration.txt",
        "latest_train_state.pt",
    }


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        ("latest_train_state.pt", "missing or empty"),
        ("iter_0000001/.metadata", "missing or empty"),
        ("iter_0000001/common.pt", "missing or empty"),
        ("iter_0000001/train_state.pt", "missing or empty"),
        ("iter_0000001/tokenizer/tokenizer.json", "missing or empty"),
        ("iter_0000001/__0_0.distcp", "no distributed checkpoint shards"),
    ],
)
def test_validate_checkpoint_rejects_incomplete_layout(tmp_path, relative_path, message):
    module = _load_module()
    checkpoint = _complete_checkpoint(module, tmp_path)
    (checkpoint / relative_path).unlink()

    with pytest.raises(ValueError, match=message):
        module.validate_checkpoint(checkpoint)


def test_validate_checkpoint_rejects_wrong_model_configuration(tmp_path):
    module = _load_module()
    checkpoint = _complete_checkpoint(module, tmp_path)
    run_config = checkpoint / "iter_0000001" / "run_config.yaml"
    run_config.write_text(run_config.read_text(encoding="utf-8").replace("seq_length: 8192", "seq_length: 4096"))

    with pytest.raises(ValueError, match="model.seq_length.*8192.*4096"):
        module.validate_checkpoint(checkpoint)


def test_validate_checkpoint_rejects_wrong_distributed_checkpoint_format(tmp_path):
    module = _load_module()
    checkpoint = _complete_checkpoint(module, tmp_path)
    metadata_path = checkpoint / "iter_0000001" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["sharded_backend"] = "zarr"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="expected torch_dist v1 layout"):
        module.validate_checkpoint(checkpoint)


def test_validate_checkpoint_rejects_wrong_tokenizer(tmp_path):
    module = _load_module()
    checkpoint = _complete_checkpoint(module, tmp_path)
    tokenizer_path = checkpoint / "iter_0000001" / "tokenizer" / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["model"]["vocab"]["<EOS>"] = 7
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer is incompatible"):
        module.validate_checkpoint(checkpoint)


def test_validate_checkpoint_rejects_legacy_output_without_provenance(tmp_path):
    module = _load_module()
    checkpoint = _write_checkpoint(module, tmp_path)

    with pytest.raises(ValueError, match="missing NVFlare conversion provenance.*legacy output"):
        module.validate_checkpoint(checkpoint)


@pytest.mark.parametrize("mutation", ["changed artifact", "extra artifact", "changed provenance"])
def test_validate_checkpoint_rejects_stale_or_unrelated_output(tmp_path, mutation):
    module = _load_module()
    checkpoint = _complete_checkpoint(module, tmp_path)
    if mutation == "changed artifact":
        (checkpoint / "iter_0000001" / "common.pt").write_bytes(b"different common state")
    elif mutation == "extra artifact":
        (checkpoint / "unrelated.pt").write_bytes(b"unrelated")
    else:
        provenance_path = checkpoint / module.PROVENANCE_FILE
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["source_model_tag"] = "evo2/different-model:1.0"
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance or artifact inventory is incompatible"):
        module.validate_checkpoint(checkpoint)

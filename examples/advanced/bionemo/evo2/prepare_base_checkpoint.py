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
"""Download Evo2-1B and atomically convert it to a Megatron Bridge checkpoint."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shlex
import shutil
import uuid
from pathlib import Path
from typing import Any

MODEL_TAG = "evo2/1b-8k-bf16:1.0"
MODEL_SIZE = "evo2_1b_base"
MIXED_PRECISION_RECIPE = "bf16_mixed"
SEQUENCE_LENGTH = 8192
CHECKPOINT_ITERATION = 1
PROVENANCE_FILE = "nvflare_conversion_provenance.json"
PROVENANCE_VERSION = 1

EXPECTED_RUN_CONFIG = {
    "checkpoint.ckpt_format": "torch_dist",
    "model._target_": "bionemo.evo2.models.evo2_provider.Hyena1bModelProvider",
    "model.bf16": True,
    "model.ffn_hidden_size": 5120,
    "model.hidden_size": 1920,
    "model.num_attention_heads": 15,
    "model.num_layers": 25,
    "model.seq_length": SEQUENCE_LENGTH,
    "model.tokenizer_library": "byte-level",
    "model.vocab_size": 512,
}
EXPECTED_CHECKPOINT_METADATA = {
    "common_backend": "torch",
    "common_backend_version": 1,
    "sharded_backend": "torch_dist",
    "sharded_backend_version": 1,
}
EXPECTED_TOKENIZER_CONFIG = {
    "backend": "tokenizers",
    "bos_token": "<BOS>",
    "cls_token": "<BOS>",
    "eos_token": "<EOS>",
    "pad_token": "<PAD>",
    "sep_token": "<SEP>",
    "tokenizer_class": "TokenizersBackend",
    "unk_token": "<UNK>",
}
EXPECTED_SPECIAL_TOKEN_IDS = {"<EOS>": 0, "<PAD>": 1, "<BOS>": 2, "<SEP>": 3, "<UNK>": 4}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Megatron Bridge checkpoint contains invalid JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Megatron Bridge checkpoint JSON must contain an object: {path}.")
    return payload


def _parse_yaml_scalar(raw_value: str) -> Any:
    """Parse the scalar forms used by the generated run_config without importing PyYAML."""

    value = raw_value.strip()
    if value.startswith(("'", '"')):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    return value


def _load_yaml_scalars(path: Path) -> dict[str, Any]:
    """Read dotted scalar paths from the simple mapping emitted by Megatron Bridge."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Megatron Bridge checkpoint contains an unreadable run config: {path}.") from exc

    values: dict[str, Any] = {}
    parents: list[tuple[int, str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped in {"---", "..."}:
            continue
        indentation = len(line) - len(line.lstrip(" "))
        content = line[indentation:]
        if ":" not in content:
            continue
        key, raw_value = content.split(":", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key):
            continue
        while parents and parents[-1][0] >= indentation:
            parents.pop()
        dotted_path = ".".join([parent_key for _, parent_key in parents] + [key])
        raw_value = raw_value.strip()
        if raw_value:
            values[dotted_path] = _parse_yaml_scalar(raw_value)
        else:
            parents.append((indentation, key))
    return values


def _require_nonempty_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Megatron Bridge checkpoint file is missing or empty: {path}.")


def _validate_layout(checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    latest_file = checkpoint / "latest_checkpointed_iteration.txt"
    _require_nonempty_file(latest_file)
    try:
        iteration = int(latest_file.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Megatron Bridge checkpoint has an invalid iteration marker: {latest_file}.") from exc
    if iteration != CHECKPOINT_ITERATION:
        raise ValueError(
            f"Megatron Bridge checkpoint iteration must be {CHECKPOINT_ITERATION}, observed {iteration}: {latest_file}."
        )

    iteration_dir = checkpoint / f"iter_{iteration:07d}"
    if not iteration_dir.is_dir():
        raise ValueError(f"Megatron Bridge checkpoint iteration is missing: {iteration_dir}.")

    required_files = [
        checkpoint / "latest_train_state.pt",
        iteration_dir / ".metadata",
        iteration_dir / "common.pt",
        iteration_dir / "metadata.json",
        iteration_dir / "run_config.yaml",
        iteration_dir / "train_state.pt",
        iteration_dir / "tokenizer" / "tokenizer.json",
        iteration_dir / "tokenizer" / "tokenizer_config.json",
    ]
    for required_file in required_files:
        _require_nonempty_file(required_file)

    shards = sorted(iteration_dir.glob("*.distcp"))
    if not shards:
        raise ValueError(f"Megatron Bridge checkpoint has no distributed checkpoint shards in {iteration_dir}.")
    for shard in shards:
        _require_nonempty_file(shard)

    metadata = _load_json(iteration_dir / "metadata.json")
    if metadata != EXPECTED_CHECKPOINT_METADATA:
        raise ValueError(
            f"Megatron Bridge checkpoint metadata does not match the expected torch_dist v1 layout: "
            f"{iteration_dir / 'metadata.json'}."
        )

    run_config = _load_yaml_scalars(iteration_dir / "run_config.yaml")
    mismatched_config = {
        key: {"expected": expected, "actual": run_config.get(key)}
        for key, expected in EXPECTED_RUN_CONFIG.items()
        if run_config.get(key) != expected
    }
    if mismatched_config:
        raise ValueError(f"Megatron Bridge run config is incompatible: {mismatched_config}.")

    tokenizer_config = _load_json(iteration_dir / "tokenizer" / "tokenizer_config.json")
    mismatched_tokenizer = {
        key: {"expected": expected, "actual": tokenizer_config.get(key)}
        for key, expected in EXPECTED_TOKENIZER_CONFIG.items()
        if tokenizer_config.get(key) != expected
    }
    tokenizer = _load_json(iteration_dir / "tokenizer" / "tokenizer.json")
    tokenizer_model = tokenizer.get("model")
    if not isinstance(tokenizer_model, dict):
        tokenizer_model = {}
    vocabulary = tokenizer_model.get("vocab")
    if (
        tokenizer.get("version") != "1.0"
        or tokenizer_model.get("type") != "WordLevel"
        or not isinstance(vocabulary, dict)
        or len(vocabulary) != 512
        or any(vocabulary.get(token) != token_id for token, token_id in EXPECTED_SPECIAL_TOKEN_IDS.items())
    ):
        mismatched_tokenizer["tokenizer.json"] = "expected the pinned 512-token byte-level WordLevel tokenizer"
    if mismatched_tokenizer:
        raise ValueError(f"Megatron Bridge tokenizer is incompatible: {mismatched_tokenizer}.")

    return iteration_dir, tokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_inventory(checkpoint: Path) -> list[dict[str, Any]]:
    inventory = []
    provenance_path = checkpoint / PROVENANCE_FILE
    for artifact in sorted(checkpoint.rglob("*")):
        if artifact.is_symlink():
            raise ValueError(f"Megatron Bridge checkpoint must not contain symbolic links: {artifact}.")
        if not artifact.is_file() or artifact == provenance_path:
            continue
        inventory.append(
            {
                "path": artifact.relative_to(checkpoint).as_posix(),
                "sha256": _sha256(artifact),
                "size": artifact.stat().st_size,
            }
        )
    return inventory


def _conversion_provenance(checkpoint: Path, tokenizer: dict[str, Any]) -> dict[str, Any]:
    inventory = _artifact_inventory(checkpoint)
    inventory_by_path = {entry["path"]: entry for entry in inventory}
    iteration = int((checkpoint / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8").strip())
    tokenizer_root = f"iter_{iteration:07d}/tokenizer"
    tokenizer_assets = {
        name: inventory_by_path[f"{tokenizer_root}/{name}"]["sha256"]
        for name in ("tokenizer.json", "tokenizer_config.json")
    }
    return {
        "conversion": {
            "mixed_precision_recipe": MIXED_PRECISION_RECIPE,
            "model_size": MODEL_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
        },
        "inventory": inventory,
        "schema_version": PROVENANCE_VERSION,
        "source_model_tag": MODEL_TAG,
        "tokenizer": {
            "assets_sha256": tokenizer_assets,
            "library": "byte-level",
            "model_type": tokenizer["model"]["type"],
            "special_token_ids": EXPECTED_SPECIAL_TOKEN_IDS,
            "vocab_size": len(tokenizer["model"]["vocab"]),
        },
    }


def _write_conversion_provenance(checkpoint: Path) -> None:
    _, tokenizer = _validate_layout(checkpoint)
    provenance = _conversion_provenance(checkpoint, tokenizer)
    (checkpoint / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_checkpoint(path: str | Path) -> Path:
    """Return ``path`` after validating the pinned conversion and its complete inventory."""

    checkpoint = Path(path).resolve()
    _, tokenizer = _validate_layout(checkpoint)
    provenance_path = checkpoint / PROVENANCE_FILE
    if not provenance_path.is_file():
        raise ValueError(
            f"Megatron Bridge checkpoint is missing NVFlare conversion provenance: {provenance_path}. "
            "Remove this legacy output and rerun checkpoint preparation."
        )
    provenance = _load_json(provenance_path)
    expected_provenance = _conversion_provenance(checkpoint, tokenizer)
    if provenance != expected_provenance:
        raise ValueError(
            f"Megatron Bridge checkpoint provenance or artifact inventory is incompatible: {provenance_path}. "
            "Remove the output and rerun checkpoint preparation."
        )
    return checkpoint


def prepare_checkpoint(output: str | Path) -> Path:
    """Download and convert the pinned checkpoint, publishing it only after validation."""

    destination = Path(output).resolve()
    if destination.exists():
        return validate_checkpoint(destination)

    try:
        from bionemo.common.data.load import load
        from bionemo.common.utils.subprocess_utils import run_subprocess_safely
        from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH_512
    except ImportError as exc:
        raise RuntimeError("Checkpoint preparation must run in the pinned Evo2 BioNeMo environment.") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial-{uuid.uuid4().hex}")
    source = load(MODEL_TAG)
    command = shlex.join(
        [
            "evo2_convert_nemo2_to_mbridge",
            "--nemo2-ckpt-dir",
            str(source),
            "--mbridge-ckpt-dir",
            str(partial),
            "--model-size",
            MODEL_SIZE,
            "--mixed-precision-recipe",
            MIXED_PRECISION_RECIPE,
            "--seq-length",
            str(SEQUENCE_LENGTH),
            "--tokenizer-path",
            str(DEFAULT_HF_TOKENIZER_MODEL_PATH_512),
        ]
    )
    try:
        result = run_subprocess_safely(command)
        if result.get("returncode") != 0 or result.get("error"):
            raise RuntimeError(
                f"Evo2 checkpoint conversion failed: {result.get('error', 'non-zero return')} "
                f"(returncode={result.get('returncode')})."
            )
        _write_conversion_provenance(partial)
        partial.replace(destination)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return validate_checkpoint(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="./models/evo2_1b_bf16_mbridge")
    args = parser.parse_args()
    checkpoint = prepare_checkpoint(args.output)
    print(f"Megatron Bridge checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

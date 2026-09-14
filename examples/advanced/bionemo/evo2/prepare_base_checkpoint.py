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
import shlex
import shutil
import uuid
from pathlib import Path

MODEL_TAG = "evo2/1b-8k-bf16:1.0"
MODEL_SIZE = "evo2_1b_base"


def validate_checkpoint(path: str | Path) -> Path:
    """Return ``path`` after checking the Megatron Bridge completion markers."""

    checkpoint = Path(path).resolve()
    latest_file = checkpoint / "latest_checkpointed_iteration.txt"
    if not latest_file.is_file():
        raise ValueError(f"Megatron Bridge checkpoint is missing {latest_file}.")
    try:
        iteration = int(latest_file.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise ValueError(f"Megatron Bridge checkpoint has an invalid iteration marker: {latest_file}.") from exc
    iteration_dir = checkpoint / f"iter_{iteration:07d}"
    if not iteration_dir.is_dir() or not any(iteration_dir.iterdir()):
        raise ValueError(f"Megatron Bridge checkpoint iteration is missing or empty: {iteration_dir}.")
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
            "bf16_mixed",
            "--seq-length",
            "8192",
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
        validate_checkpoint(partial)
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

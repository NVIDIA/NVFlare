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

"""One-command MNIST FedCoRe workflow."""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from src.data import MNISTDataConfig, validate_mnist_config
from src.validation import (
    non_negative_float,
    non_negative_int,
    parse_alpha_grid,
    positive_float,
    positive_int,
    probability,
)

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]


def _run(command: list[str], cwd: Path = PROJECT_DIR) -> None:
    print("\n$ " + shlex.join(command), flush=True)
    env = os.environ.copy()
    # Keep external simulator clients in the active virtual environment. Resolving a venv's Python symlink here can
    # replace its bin directory with the base interpreter directory before clients execute portable `python3` commands.
    python_bin = str(Path(sys.executable).parent)
    env["PATH"] = os.pathsep.join([python_bin, env.get("PATH", "")])
    # Explicit --workspace values must remain authoritative so checkpoint discovery uses the same tree as SimEnv.
    env.pop("NVFLARE_SIMULATOR_WORKSPACE_ROOT", None)
    subprocess.run(command, cwd=cwd, check=True, env=env)


def _first_gpu(gpu: str | None) -> str:
    matches = re.findall(r"\d+", gpu or "")
    return matches[0] if matches else "0"


def _latest_checkpoint(workspace: Path) -> Path:
    preferred = list(workspace.rglob("FL_global_model.pt"))
    candidates = preferred or list(workspace.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No NVFlare model checkpoint was found under {workspace}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _prepare_run_directories(output_dir: Path, workspace: Path) -> None:
    """Reserve fresh output and workspace directories so runs cannot share artifacts."""

    if output_dir == workspace:
        raise ValueError("FedCoRe output and workspace directories must be different paths.")
    paths = ((output_dir, "output directory"), (workspace, "NVFlare workspace"))
    for path, description in paths:
        if path.exists():
            raise FileExistsError(
                f"FedCoRe {description} already exists: {path}. Choose a fresh path to avoid stale or concurrent artifacts."
            )
    output_dir.mkdir(parents=True, exist_ok=False)
    workspace.mkdir(parents=True, exist_ok=False)


def _validate_run_configuration(args, proxy_strength: float) -> None:
    validate_mnist_config(
        MNISTDataConfig(
            output_dir=Path("."),
            dataset_root=Path("."),
            scenario=args.scenario,
            train_samples_per_site=args.train_samples_per_site,
            val_samples_per_site=args.val_samples_per_site,
            test_samples_per_site=args.test_samples_per_site,
            proxy_strength=proxy_strength,
            seed=args.seed,
        )
    )
    parse_alpha_grid(args.alpha_grid)


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the public MNIST FedCoRe starter end to end.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--scenario", choices=["recoverable", "uninformative"], default="recoverable")
    parser.add_argument("--proxy-strength", type=probability, default=None)
    parser.add_argument("--dataset-root", default="~/.cache/nvflare/fedcore")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument(
        "--gpu",
        default=None,
        help='GPU allocation. Quick mode defaults to GPU 0; full mode defaults to one GPU per client, "[0],[1],[2]".',
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--workspace", default="")
    parser.add_argument("--seed", type=non_negative_int, default=7)
    parser.add_argument("--train-samples-per-site", type=positive_int, default=96)
    parser.add_argument("--val-samples-per-site", type=positive_int, default=64)
    parser.add_argument("--test-samples-per-site", type=positive_int, default=64)
    parser.add_argument("--feature-batch-size", type=positive_int, default=2)
    parser.add_argument("--num-rounds", type=positive_int, default=10)
    parser.add_argument("--local-epochs", type=positive_int, default=10)
    parser.add_argument("--hidden-dim", type=positive_int, default=128)
    parser.add_argument("--learning-rate", type=positive_float, default=1e-3)
    parser.add_argument("--task-weight", type=non_negative_float, default=4.0)
    parser.add_argument("--effect-weight", type=non_negative_float, default=0.25)
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1,1.5,2")
    parser.add_argument("--predictor-rounds", type=positive_int, default=1)
    parser.add_argument("--predictor-max-steps", type=positive_int, default=10)
    parser.add_argument("--lora-r", type=positive_int, default=64)
    parser.add_argument("--lora-alpha", type=positive_int, default=128)
    return parser


def _build_predictor_command(args, qwen_example: Path, data_dir: Path, predictor_workspace: Path) -> list[str]:
    command = [
        sys.executable,
        str(qwen_example / "job.py"),
        "--n_clients",
        "3",
        "--num_rounds",
        str(args.predictor_rounds),
        "--data_dir",
        str(data_dir),
        "--image_root",
        str(data_dir),
        "--model_name_or_path",
        args.model_name_or_path,
        "--max_steps",
        str(args.predictor_max_steps),
        "--lora",
        "--lora_r",
        str(args.lora_r),
        "--lora_alpha",
        str(args.lora_alpha),
        "--workspace",
        str(predictor_workspace),
    ]
    if args.gpu:
        command.extend(["--gpu", args.gpu])
    return command


def main() -> None:
    args = define_parser().parse_args()
    if args.scenario == "uninformative" and args.proxy_strength is not None:
        raise ValueError("--proxy-strength is fixed at 0.5 for the uninformative scenario.")
    proxy_strength = (
        args.proxy_strength if args.proxy_strength is not None else (0.9 if args.scenario == "recoverable" else 0.5)
    )
    _validate_run_configuration(args, proxy_strength)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (PROJECT_DIR / "outputs" / f"{args.scenario}_seed{args.seed}").resolve()
    )
    workspace = Path(args.workspace).expanduser().resolve() if args.workspace else (output_dir / "workspace").resolve()
    data_dir = output_dir / "data"
    cache_dir = output_dir / "feature_cache"
    completion_output = output_dir / "completion"
    evaluation_output = output_dir / "evaluation"
    _prepare_run_directories(output_dir, workspace)
    run_config = vars(args).copy()
    run_config.update(
        {
            "resolved_output_dir": str(output_dir),
            "resolved_workspace": str(workspace),
            "resolved_dataset_root": str(dataset_root),
            "resolved_proxy_strength": proxy_strength,
        }
    )
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")

    prepare_command = [
        sys.executable,
        str(PROJECT_DIR / "prepare_data.py"),
        "--output-dir",
        str(data_dir),
        "--dataset-root",
        str(dataset_root),
        "--scenario",
        args.scenario,
        "--train-samples-per-site",
        str(args.train_samples_per_site),
        "--val-samples-per-site",
        str(args.val_samples_per_site),
        "--test-samples-per-site",
        str(args.test_samples_per_site),
        "--seed",
        str(args.seed),
    ]
    if args.scenario == "recoverable":
        prepare_command.extend(["--proxy-strength", str(proxy_strength)])
    _run(prepare_command)

    adapter_checkpoint = ""
    if args.mode == "full":
        qwen_example = REPO_ROOT / "examples" / "advanced" / "qwen3-vl"
        predictor_workspace = workspace / "qwen_predictor"
        predictor_command = _build_predictor_command(args, qwen_example, data_dir, predictor_workspace)
        _run(predictor_command, cwd=qwen_example)
        adapter_checkpoint = str(_latest_checkpoint(predictor_workspace))
        print(f"Using federated Qwen LoRA checkpoint: {adapter_checkpoint}")

    cache_command = [
        sys.executable,
        str(PROJECT_DIR / "cache_features.py"),
        "--data-dir",
        str(data_dir),
        "--cache-dir",
        str(cache_dir),
        "--model-name-or-path",
        args.model_name_or_path,
        "--device",
        f"cuda:{_first_gpu(args.gpu)}",
        "--batch-size",
        str(args.feature_batch_size),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
    ]
    if adapter_checkpoint:
        cache_command.extend(["--adapter-checkpoint", adapter_checkpoint])
    _run(cache_command)

    completion_workspace = workspace / "completion"
    _run(
        [
            sys.executable,
            str(PROJECT_DIR / "job.py"),
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(completion_output),
            "--workspace",
            str(completion_workspace),
            "--num-rounds",
            str(args.num_rounds),
            "--local-epochs",
            str(args.local_epochs),
            "--hidden-dim",
            str(args.hidden_dim),
            "--learning-rate",
            str(args.learning_rate),
            "--task-weight",
            str(args.task_weight),
            "--effect-weight",
            str(args.effect_weight),
            "--seed",
            str(args.seed),
        ]
    )
    completion_checkpoint = _latest_checkpoint(completion_workspace)
    published_checkpoint = completion_output / "global_model.pt"
    shutil.copy2(completion_checkpoint, published_checkpoint)
    _run(
        [
            sys.executable,
            str(PROJECT_DIR / "evaluate.py"),
            "--cache-dir",
            str(cache_dir),
            "--checkpoint",
            str(published_checkpoint),
            "--output-dir",
            str(evaluation_output),
            "--hidden-dim",
            str(args.hidden_dim),
            "--alpha-grid",
            args.alpha_grid,
        ]
    )
    print("\nFedCoRe workflow complete")
    print(f"  output directory: {output_dir}")
    print(f"  result summary:   {evaluation_output / 'summary.json'}")


if __name__ == "__main__":
    main()

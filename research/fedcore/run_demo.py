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

"""One-command synthetic FedCoRe workflow."""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]


def _run(command: list[str], cwd: Path = PROJECT_DIR) -> None:
    print("\n$ " + shlex.join(command), flush=True)
    env = os.environ.copy()
    python_bin = str(Path(sys.executable).resolve().parent)
    env["PATH"] = os.pathsep.join([python_bin, env.get("PATH", "")])
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


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the public FedCoRe synthetic starter end to end.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--scenario", choices=["recoverable", "uninformative"], default="recoverable")
    parser.add_argument("--proxy-strength", type=float, default=None)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--feature-backend", choices=["qwen", "mock"], default="qwen")
    parser.add_argument(
        "--gpu",
        default=None,
        help='GPU allocation. Quick mode defaults to GPU 0; full mode defaults to one GPU per client, "[0],[1],[2]".',
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--workspace", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-samples-per-site", type=int, default=48)
    parser.add_argument("--val-samples-per-site", type=int, default=16)
    parser.add_argument("--test-samples-per-site", type=int, default=16)
    parser.add_argument("--feature-batch-size", type=int, default=2)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1,1.5,2")
    parser.add_argument("--predictor-rounds", type=int, default=1)
    parser.add_argument("--predictor-max-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
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
    if args.mode == "full" and args.feature_backend != "qwen":
        raise ValueError("Full mode requires --feature-backend qwen.")
    proxy_strength = args.proxy_strength
    if proxy_strength is None:
        proxy_strength = 0.9 if args.scenario == "recoverable" else 0.5
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (PROJECT_DIR / "outputs" / f"{args.scenario}_seed{args.seed}").resolve()
    )
    workspace = (
        Path(args.workspace).expanduser().resolve()
        if args.workspace
        else (Path("/tmp/nvflare/fedcore") / f"{args.scenario}_seed{args.seed}").resolve()
    )
    data_dir = output_dir / "data"
    cache_dir = output_dir / "feature_cache"
    completion_output = output_dir / "completion"
    evaluation_output = output_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    run_config.update(
        {
            "resolved_output_dir": str(output_dir),
            "resolved_workspace": str(workspace),
            "resolved_proxy_strength": proxy_strength,
        }
    )
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")

    _run(
        [
            sys.executable,
            str(PROJECT_DIR / "prepare_data.py"),
            "--output-dir",
            str(data_dir),
            "--train-samples-per-site",
            str(args.train_samples_per_site),
            "--val-samples-per-site",
            str(args.val_samples_per_site),
            "--test-samples-per-site",
            str(args.test_samples_per_site),
            "--proxy-strength",
            str(proxy_strength),
            "--seed",
            str(args.seed),
        ]
    )

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
        "--backend",
        args.feature_backend,
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

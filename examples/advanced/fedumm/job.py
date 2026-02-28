# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""FedJob API configuration for federated VLM fine-tuning.

Supports two environment isolation modes:

1. **In-process** (default, for simulator / same-env):
   ``python job.py --model_backend blip_vqa --simulator``

2. **External-process with conda env** (for production / env isolation):
   ``python job.py --model_backend januspro --use_env_script --export_dir ./jobs/januspro``

When ``--use_env_script`` is set, NVFlare's ScriptRunner launches the
training script via ``bash scripts/launch_<backend>.sh`` which activates
the matching conda environment before executing ``python -u fl_client.py``.
This avoids dependency conflicts between BLIP (transformers only) and
JanusPro (requires the ``janus`` package with custom model code).
"""

import argparse
import os

from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner


def _parse_args():
    p = argparse.ArgumentParser(description="Generate or simulate a VLM FL job")
    # --- model ---
    p.add_argument("--model_backend", type=str, default="blip_vqa",
                    choices=["blip_vqa", "januspro"],
                    help="Which VLM backend to use.")
    p.add_argument("--model_name_or_path", type=str, default="",
                    help="HF model id (uses backend default if empty).")
    # --- FL ---
    p.add_argument("--num_clients", type=int, default=2)
    p.add_argument("--num_rounds", type=int, default=3)
    p.add_argument("--local_epochs", type=int, default=1)
    # --- training ---
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--max_eval_samples", type=int, default=-1)
    p.add_argument("--data_path", type=str, default="")
    p.add_argument("--dirichlet_alpha", type=float, default=0.0,
                    help="Dirichlet concentration for non-IID data partition. "
                         "0 = IID, 0.1/0.5/1.0 = non-IID levels from paper.")
    p.add_argument("--seed", type=int, default=42)
    # --- environment ---
    p.add_argument("--use_env_script", action="store_true",
                    help="Launch via bash wrapper that activates a conda "
                         "environment (enables env isolation between models).")
    # --- output ---
    p.add_argument("--export_dir", type=str, default="",
                    help="Export job config to this directory (for production).")
    p.add_argument("--simulator", action="store_true",
                    help="Run with the NVFlare simulator.")
    return p.parse_args()


# Map backend name -> launcher wrapper script
_LAUNCH_SCRIPTS = {
    "blip_vqa": "scripts/launch_blip.sh",
    "januspro": "scripts/launch_januspro.sh",
}


def main() -> None:
    args = _parse_args()

    job = FedJob(name=f"vlm_fedavg_{args.model_backend}")

    # ---- Server ----
    controller = FedAvg(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
    )
    job.to(controller, "server")
    job.to(IntimeModelSelector(key_metric="global_acc", negate=False), "server")

    # ---- Build script_args ----
    script_args = (
        f" --model_backend {args.model_backend}"
        f" --num_clients {args.num_clients}"
        f" --local_epochs {args.local_epochs}"
        f" --batch_size {args.batch_size}"
        f" --grad_accum {args.grad_accum}"
        f" --lr {args.lr}"
        f" --lora_r {args.lora_r}"
        f" --lora_alpha {args.lora_alpha}"
        f" --max_train_samples {args.max_train_samples}"
        f" --max_eval_samples {args.max_eval_samples}"
        f" --dirichlet_alpha {args.dirichlet_alpha}"
        f" --seed {args.seed}"
    )
    if args.model_name_or_path:
        script_args += f" --model_name_or_path {args.model_name_or_path}"
    if args.data_path:
        script_args += f" --data_path {args.data_path}"

    # ---- Clients ----
    for i in range(args.num_clients):
        site = f"site-{i + 1}"

        if args.use_env_script:
            # External-process mode: launch via bash wrapper
            # The wrapper activates the correct conda env, then runs python
            launch_script = _LAUNCH_SCRIPTS[args.model_backend]
            runner = ScriptRunner(
                script="src/fl_client.py",
                script_args=script_args,
                launch_external_process=True,
                command=f"bash {launch_script}",
            )
        else:
            # In-process mode: same python environment (simulator / dev)
            runner = ScriptRunner(
                script="src/fl_client.py",
                script_args=script_args,
                launch_external_process=False,
            )

        job.to(runner, site)

    # ---- Execute ----
    if args.export_dir:
        job.export_job(args.export_dir)
        print(f"Job exported to: {args.export_dir}")
    elif args.simulator:
        job.simulator_run(
            os.path.join(os.getcwd(), "workspace_simulator"),
            n_clients=args.num_clients,
            gpu="0",
        )
    else:
        print("Specify --export_dir or --simulator.")
        print("Examples:")
        print(f"  python job.py --model_backend {args.model_backend} --simulator")
        print(f"  python job.py --model_backend {args.model_backend} "
              f"--use_env_script --export_dir ./jobs/{args.model_backend}")


if __name__ == "__main__":
    main()

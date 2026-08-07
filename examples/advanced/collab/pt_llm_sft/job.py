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

"""Run full-parameter federated SFT with Hugging Face and the Collab API."""

import argparse
from pathlib import Path

from client import LLMSFTClient
from server import SFTFedAvg

from nvflare.collab import CollabRecipe, simple_logging
from nvflare.recipe import SimEnv

DEFAULT_DATA_ROOT = "/tmp/nvflare/collab/pt_llm_sft/data"
DEFAULT_OUTPUT_ROOT = "/tmp/nvflare/collab/pt_llm_sft/results"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
EXAMPLE_DIR = Path(__file__).resolve().parent


def make_recipe(args) -> CollabRecipe:
    client = LLMSFTClient(
        model_name_or_path=args.model_name_or_path,
        data_root=str(args.data_root),
        output_root=str(args.output_root),
        syncs_per_epoch=args.syncs_per_epoch,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        evaluate_global_model=not args.skip_evaluation,
        model_revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        precision=args.precision,
    )
    server = SFTFedAvg(
        num_epochs=args.num_epochs,
        syncs_per_epoch=args.syncs_per_epoch,
        min_clients=args.num_clients,
        output_root=str(args.output_root),
        call_timeout=args.call_timeout,
        save_every_sync=args.save_every_sync,
    )

    # CollabRecipe wires the @collab.main server object to one client object copied to every site.
    recipe = CollabRecipe(
        job_name="pt_llm_sft",
        server=server,
        client=client,
        min_clients=args.num_clients,
        sync_task_timeout=args.call_timeout,
    )
    # The recipe ships client.py automatically; add its model.py dependency to every client app.
    recipe.add_client_file(str(EXAMPLE_DIR / "model.py"))
    return recipe


def make_env(args) -> SimEnv:
    """Create the local multi-client simulator environment used by the recipe."""
    return SimEnv(
        num_clients=args.num_clients,
        gpu_config=args.gpu_config,
        workspace_root=args.workspace_root,
    )


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workspace-root", default="/tmp/nvflare/collab/pt_llm_sft/workspace")
    parser.add_argument(
        "--gpu-config",
        help='Simulator GPU assignment, for example "0,1,2,3" for one GPU per client',
    )
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow code from the model repository to execute; only use with a trusted checkpoint",
    )
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--syncs-per-epoch", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--precision", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--call-timeout", type=float, default=1800)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--save-every-sync", action="store_true")
    return parser


def main() -> None:
    args = define_parser().parse_args()
    if args.num_clients < 1 or args.num_epochs < 1 or args.syncs_per_epoch < 1:
        raise ValueError("--num-clients, --num-epochs, and --syncs-per-epoch must be at least 1")

    args.data_root = Path(args.data_root).expanduser().resolve()
    args.output_root = Path(args.output_root).expanduser().resolve()

    simple_logging()
    print(
        "Starting Collab full-parameter SFT simulation\n"
        f"  model: {args.model_name_or_path}\n"
        f"  clients: {args.num_clients}\n"
        f"  epochs: {args.num_epochs}\n"
        f"  syncs per epoch: {args.syncs_per_epoch}\n"
        f"  GPU config: {args.gpu_config or 'not set'}\n"
        f"  data: {args.data_root}"
    )

    # execute finalizes the job and runs or exports it in the requested SimEnv.
    run = make_recipe(args).execute(make_env(args))
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()

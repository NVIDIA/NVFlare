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

import argparse
import os
import sys

FEDBPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(FEDBPT_DIR, os.pardir, os.pardir))
DEFAULT_WORKSPACE = "/tmp/nvflare/fedbpt"

if FEDBPT_DIR not in sys.path:
    sys.path.insert(0, FEDBPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def define_parser():
    parser = argparse.ArgumentParser(description="Run or export the FedBPT NVFlare job.")

    parser.add_argument("--job_name", default="fedbpt", help="Name of the generated NVFlare job.")
    parser.add_argument("--num_clients", type=int, default=10, help="Number of FedBPT clients.")
    parser.add_argument("--min_clients", type=int, default=None, help="Minimum clients required to start the job.")
    parser.add_argument("--num_rounds", type=int, default=200, help="Number of global FedBPT rounds.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed used by the server and client scripts.")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="Simulation workspace root.")
    parser.add_argument("--threads", type=int, default=None, help="Number of simulator worker threads.")
    parser.add_argument("--gpu", default=None, help="Simulator GPU assignment string, for example '0,1,2,3'.")
    parser.add_argument("--log_config", default=None, help="Simulator log config mode or path.")
    parser.add_argument("--export", action="store_true", help="Export the job and exit without running simulation.")
    parser.add_argument("--export-dir", default="./jobs", help="Directory where the job folder is exported.")

    parser.add_argument("--frac", type=float, default=1.0, help="Fraction of clients used by GlobalES.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Initial GlobalES sigma.")
    parser.add_argument("--intrinsic_dim", type=int, default=500, help="Intrinsic dimension for GlobalES and clients.")
    parser.add_argument("--bound", type=int, default=0, help="GlobalES bound, 0 disables bounds.")

    parser.add_argument("--task_name", default="sst2", help="Task name passed to fedbpt_train.py.")
    parser.add_argument("--n_prompt_tokens", type=int, default=50, help="Number of prompt tokens.")
    parser.add_argument("--k_shot", type=int, default=200, help="Few-shot samples per class.")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional client batch size override.")
    parser.add_argument("--device", default="cuda:0", help="Device passed to fedbpt_train.py.")
    parser.add_argument("--loss_type", default="ce", help="Loss type passed to fedbpt_train.py.")
    parser.add_argument("--cat_or_add", default="add", help="Prompt mode passed to fedbpt_train.py.")
    parser.add_argument("--local_iter", type=int, default=8, help="Local CMA iterations per round.")
    parser.add_argument("--num_users", type=int, default=None, help="Number of client data shards.")
    parser.add_argument("--iid", type=int, default=1, help="Whether to split client data IID.")
    parser.add_argument("--local_popsize", type=int, default=5, help="Local CMA population size.")
    parser.add_argument("--perturb", type=int, default=1, help="Whether to use perturbed data fitness.")
    parser.add_argument("--model_name", default="roberta-large", choices=["roberta-base", "roberta-large"])
    parser.add_argument("--eval_clients", default="site-1", help="Comma-separated clients that run global eval.")
    parser.add_argument("--llama_causal", type=int, default=1, help="FedBPT compatibility flag.")
    parser.add_argument(
        "--train_args",
        default="",
        help="Additional fedbpt_train.py arguments appended after generated arguments.",
    )
    return parser


def create_recipe(args, extra_train_args=None):
    # Import only after the entrypoint has parsed its own export flags. Recipe's
    # module initialization consumes those flags from sys.argv for execute().
    from fedbpt_recipe import FedBPTRecipe

    return FedBPTRecipe(
        name=args.job_name,
        num_clients=args.num_clients,
        min_clients=args.min_clients,
        num_rounds=args.num_rounds,
        seed=args.seed,
        frac=args.frac,
        sigma=args.sigma,
        intrinsic_dim=args.intrinsic_dim,
        bound=args.bound,
        task_name=args.task_name,
        n_prompt_tokens=args.n_prompt_tokens,
        k_shot=args.k_shot,
        batch_size=args.batch_size,
        device=args.device,
        loss_type=args.loss_type,
        cat_or_add=args.cat_or_add,
        local_iter=args.local_iter,
        num_users=args.num_users,
        iid=args.iid,
        local_popsize=args.local_popsize,
        perturb=args.perturb,
        model_name=args.model_name,
        eval_clients=args.eval_clients,
        llama_causal=args.llama_causal,
        train_args=args.train_args,
        extra_train_args=extra_train_args,
    )


def main():
    parser = define_parser()
    args, extra_train_args = parser.parse_known_args()

    recipe = create_recipe(args, extra_train_args)
    if args.export:
        recipe.export(args.export_dir)
        print(f"Job exported to: {os.path.join(args.export_dir, args.job_name)}")
        return

    from nvflare.recipe.sim_env import SimEnv

    env = SimEnv(
        num_clients=args.num_clients,
        num_threads=args.threads,
        gpu_config=args.gpu,
        log_config=args.log_config,
        workspace_root=args.workspace,
    )
    run = recipe.run(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result(clean_up=False))
    print()


if __name__ == "__main__":
    main()

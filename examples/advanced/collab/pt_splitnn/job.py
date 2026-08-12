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

"""Run two-party CIFAR-10 SplitNN training with a CollabRecipe."""

import argparse
import logging
from pathlib import Path

from client import BATCH_SIZE, CALL_TIMEOUT, NUM_STEPS, SplitNNClient
from server import SplitNNServer

from nvflare.collab import CollabRecipe, simple_logging
from nvflare.recipe import SimEnv

JOB_NAME = "collab_pt_splitnn"
DEFAULT_DATASET_ROOT = "/tmp/cifar10"
DEFAULT_PSI_WORKSPACE = "/tmp/nvflare/cifar10_psi"
EXAMPLE_DIR = Path(__file__).resolve().parent


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CIFAR-10 SplitNN training with the Collab API")
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="CIFAR-10 root populated by prepare_data.py",
    )
    parser.add_argument(
        "--psi-workspace",
        default=DEFAULT_PSI_WORKSPACE,
        help="Workspace produced by prepare_data.py",
    )
    parser.add_argument("--workspace-root", default="/tmp/nvflare/collab")
    return parser


def _intersection_file(psi_workspace: str, site_name: str) -> str:
    path = (
        Path(psi_workspace).expanduser().resolve() / site_name / "simulate_job" / site_name / "psi" / "intersection.txt"
    )
    if not path.is_file():
        raise FileNotFoundError(f"PSI intersection file not found at {path}. Run `python prepare_data.py` first.")
    return str(path)


def make_recipe(dataset_root: str, psi_workspace: str) -> CollabRecipe:
    """Connect the server, shared client code, and prepared PSI artifacts."""
    dataset_root = str(Path(dataset_root).expanduser().resolve())
    intersection_files = {site_name: _intersection_file(psi_workspace, site_name) for site_name in ("site-1", "site-2")}

    # CollabRecipe packages the server object and shared client object as an
    # NVFlare job while preserving their decorated Python call boundaries.
    recipe = CollabRecipe(
        job_name=JOB_NAME,
        server=SplitNNServer(),
        # The same client implementation is deployed to both sites.
        client=SplitNNClient(),
        min_clients=2,
        sync_task_timeout=CALL_TIMEOUT,
    )
    # Common properties are read by clients with collab.get_app_prop().
    recipe.set_client_prop("dataset_root", dataset_root)
    # Each client receives the role and intersection file produced for its site.
    recipe.set_per_site_config(
        {
            "site-1": {"role": "image", "intersection_file": intersection_files["site-1"]},
            "site-2": {"role": "label", "intersection_file": intersection_files["site-2"]},
        }
    )
    # Include modules imported by client.py in each generated client app.
    recipe.add_client_file(str(EXAMPLE_DIR / "data.py"))
    recipe.add_client_file(str(EXAMPLE_DIR / "model.py"))
    return recipe


def main():
    """Build the recipe from prepared SplitNN data artifacts and simulate it."""
    args = define_parser().parse_args()
    simple_logging(logging.INFO)
    recipe = make_recipe(args.dataset_root, args.psi_workspace)

    print("=" * 80)
    print("CIFAR-10 COLLAB SPLITNN")
    print("  Image/model-bottom site: site-1")
    print("  Label/model-top site: site-2")
    print(f"  Steps: {NUM_STEPS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Dataset root: {Path(args.dataset_root).expanduser().resolve()}")
    print(f"  PSI workspace: {Path(args.psi_workspace).expanduser().resolve()}")
    print("=" * 80)

    # The recipe is deployment-independent; SimEnv selects local simulation.
    # Quiet framework transfer logs keep output and end-to-end timing focused on the workload.
    env = SimEnv(clients=recipe.configured_sites(), workspace_root=args.workspace_root, log_config="ERROR")
    run = recipe.execute(env)
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()

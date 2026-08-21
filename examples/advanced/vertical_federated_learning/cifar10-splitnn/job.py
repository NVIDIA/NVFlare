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
from data import get_intersection_file
from server import SplitNNServer

from nvflare.collab import CollabRecipe, simple_logging
from nvflare.recipe import SimEnv

JOB_NAME = "cifar10_splitnn"
DEFAULT_DATASET_ROOT = "/tmp/cifar10"
DEFAULT_SPLIT_DIR = "/tmp/cifar10_vert_splits"
EXAMPLE_DIR = Path(__file__).resolve().parent
SITE_ROLES = {"site-1": "image", "site-2": "label"}
IMAGE_SITE = next(site_name for site_name, role in SITE_ROLES.items() if role == "image")
LABEL_SITE = next(site_name for site_name, role in SITE_ROLES.items() if role == "label")


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CIFAR-10 SplitNN training with the Collab API")
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="CIFAR-10 root populated by prepare_data.py",
    )
    parser.add_argument(
        "--split-dir",
        default=DEFAULT_SPLIT_DIR,
        help="Prepared split and intersection directory",
    )
    parser.add_argument("--workspace-root", default="/tmp/nvflare")
    return parser


def _require_intersection_file(split_dir: str, site_name: str) -> str:
    path = get_intersection_file(Path(split_dir).expanduser().resolve(), site_name)
    if not path.is_file():
        raise FileNotFoundError(f"PSI intersection file not found at {path}. Run `python prepare_data.py` first.")
    return str(path)


def make_recipe(dataset_root: str, split_dir: str) -> CollabRecipe:
    """Connect the server, shared client code, and prepared PSI artifacts."""
    dataset_root = str(Path(dataset_root).expanduser().resolve())
    intersection_files = {site_name: _require_intersection_file(split_dir, site_name) for site_name in SITE_ROLES}

    # CollabRecipe packages the server object and shared client object as an
    # NVFlare job while preserving their decorated Python call boundaries.
    recipe = CollabRecipe(
        job_name=JOB_NAME,
        server=SplitNNServer(image_site=IMAGE_SITE),
        # The same client implementation is deployed to both sites.
        client=SplitNNClient(),
        min_clients=2,
        sync_task_timeout=CALL_TIMEOUT,
    )
    # Common properties are read by clients with collab.get_app_prop().
    recipe.set_client_prop("dataset_root", dataset_root)
    recipe.set_client_prop("label_site", LABEL_SITE)
    # Each client receives the role and intersection file produced for its site.
    recipe.set_per_site_config(
        {
            site_name: {"role": role, "intersection_file": intersection_files[site_name]}
            for site_name, role in SITE_ROLES.items()
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
    recipe = make_recipe(args.dataset_root, args.split_dir)

    print("=" * 80)
    print("CIFAR-10 COLLAB SPLITNN")
    print(f"  Image/model-bottom site: {IMAGE_SITE}")
    print(f"  Label/model-top site: {LABEL_SITE}")
    print(f"  Steps: {NUM_STEPS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Dataset root: {Path(args.dataset_root).expanduser().resolve()}")
    print(f"  Prepared split directory: {Path(args.split_dir).expanduser().resolve()}")
    print("=" * 80)

    # The recipe is deployment-independent; SimEnv selects local simulation.
    # Quiet framework transfer logs keep output and end-to-end timing focused on the workload.
    env = SimEnv(clients=recipe.configured_sites(), workspace_root=args.workspace_root, log_config="ERROR")
    run = recipe.execute(env)
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()

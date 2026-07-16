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

"""Shared runtime selector for the collab examples.

Every example builds its job with a ``make_recipe()`` function and hands the
result to :func:`run_recipe`. The execution environment is an option, never a
separate example:

    --runtime in_process     threads in this process (default, fastest iteration)
    --runtime multi_process  real FLARE processes via a local POC deployment
    --runtime prod           submit to a provisioned deployment (--startup-kit)
    --runtime export         write the job definition to disk (--job-root)

The recipe is identical in all four cases; only the environment changes.
"""

import argparse
import logging

from nvflare.collab import InProcessEnv, MultiProcessEnv, simple_logging
from nvflare.recipe import ProdEnv

RUNTIMES = ("in_process", "multi_process", "prod", "export")


def make_parser(description: str) -> argparse.ArgumentParser:
    """Create an argument parser preloaded with the shared runtime options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--runtime", "-r", choices=RUNTIMES, default="in_process", help="execution environment")
    parser.add_argument("--num-clients", "-n", type=int, default=2, help="client count")
    parser.add_argument("--startup-kit", help="admin startup kit dir (required for --runtime prod)")
    parser.add_argument("--job-root", default=".", help="output dir for --runtime export")
    parser.add_argument("--log-level", default="INFO", help="python logging level")
    return parser


def run_recipe(recipe, args):
    """Run (or export) a recipe in the environment selected by --runtime."""
    simple_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    if args.runtime == "export":
        recipe.export(args.job_root)
        print(f"job exported at {args.job_root}/{recipe.job_name}")
        return None

    if args.runtime == "in_process":
        env = InProcessEnv(num_clients=args.num_clients)
        run = recipe.execute(env)
        print()
        print("Job Status:", run.get_status())
        print("Results at:", run.get_result())
        return run

    if args.runtime == "multi_process":
        env = MultiProcessEnv(num_clients=args.num_clients)
        try:
            run = recipe.execute(env)
            print()
            print("Job Status:", run.get_status())
            print("Results at:", run.get_result())
            return run
        finally:
            print("Cleaning up POC environment...")
            env.stop(clean_up=True)

    if args.runtime == "prod":
        if not args.startup_kit:
            raise SystemExit("--runtime prod requires --startup-kit pointing at the admin startup kit")
        env = ProdEnv(startup_kit_location=args.startup_kit, login_timeout=30.0)
        run = recipe.execute(env)
        print()
        print("Job Status:", run.get_status())
        print("Results at:", run.get_result())
        return run

    raise SystemExit(f"unknown runtime {args.runtime!r}")

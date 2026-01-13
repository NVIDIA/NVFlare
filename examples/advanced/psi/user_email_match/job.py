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

"""Run PSI user-email matching as an executable Python job.

This script defines and runs the PSI user-email match job directly via the
NVFlare Recipe API and SimEnv.

It can also export
the generated job configuration for use with other NVFlare runtimes.
"""

import argparse

from local_psi import LocalPSI

from nvflare.app_common.psi.recipes.dh_psi import DhPSIRecipe
from nvflare.recipe import SimEnv


def _define_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executable PSI user_email_match job (Recipe + SimEnv).")
    parser.add_argument("--n_clients", type=int, default=3, help="Number of simulated clients (site-1..site-N).")
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="/tmp/nvflare/psi/data",
        help="Root directory containing per-site data: <data_root_dir>/<site>/data.csv",
    )
    parser.add_argument(
        "--psi_output_path",
        type=str,
        default="psi/intersection.txt",
        help="Per-site output path within each client's job workspace.",
    )
    parser.add_argument(
        "--workspace_root",
        type=str,
        default="/tmp/nvflare/psi",
        help="Simulation workspace root (results will be under <workspace_root>/<job_name>/...).",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help="Simulation threads. Defaults to n_clients.",
    )
    parser.add_argument(
        "--export_only",
        action="store_true",
        help="If set, only export the job config and do not execute.",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default="/tmp/nvflare/psi/jobs/user_email_match",
        help="Directory to export the job config to when --export_only is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = _define_parser()

    local_psi = LocalPSI(data_root_dir=args.data_root_dir)

    recipe = DhPSIRecipe(
        name="user_email_match",
        min_clients=args.n_clients,
        local_psi=local_psi,
        output_path=args.psi_output_path,
    )

    if args.export_only:
        recipe.export(args.export_dir)
        print(f"Exported job config to: {args.export_dir}")
        return

    env = SimEnv(
        num_clients=args.n_clients,
        num_threads=args.num_threads,
        workspace_root=args.workspace_root,
    )
    run = recipe.execute(env)

    print()
    print("Simulation completed.")
    print("Job Status:", run.get_status())
    print("Result can be found in:", run.get_result())
    print()


if __name__ == "__main__":
    main()

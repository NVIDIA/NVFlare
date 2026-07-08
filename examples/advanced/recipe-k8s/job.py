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

from nvflare.apis.job_def import JobMetaKey
from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.recipe import ProdEnv, set_recipe_meta


def non_negative_int(value: str) -> int:
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return value


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit a NumPy FedAvg recipe to two NVFlare clients that use Kubernetes job launchers."
    )
    parser.add_argument("--startup-kit", required=True, help="Path to the production admin startup directory.")
    parser.add_argument("--username", default="admin@nvidia.com", help="Provisioned admin participant name.")
    parser.add_argument("--study", default="default", help="Study in which to submit the job.")
    parser.add_argument("--job-name", default="hello-numpy-k8s")
    parser.add_argument("--num-rounds", type=int, default=3)

    parser.add_argument("--site-1-name", default="site-1", help="First provisioned NVFlare client name.")
    parser.add_argument("--site-2-name", default="site-2", help="Second provisioned NVFlare client name.")
    parser.add_argument("--site-1-image", required=True, help="Job image pullable by the first client's cluster.")
    parser.add_argument("--site-2-image", required=True, help="Job image pullable by the second client's cluster.")
    parser.add_argument("--site-1-gpus", type=non_negative_int, default=0)
    parser.add_argument("--site-2-gpus", type=non_negative_int, default=0)
    parser.add_argument("--site-1-cpu", default="1", help="Kubernetes CPU limit for the first client job pod.")
    parser.add_argument("--site-2-cpu", default="1", help="Kubernetes CPU limit for the second client job pod.")
    parser.add_argument("--site-1-memory", default="2Gi", help="Memory limit for the first client job pod.")
    parser.add_argument("--site-2-memory", default="2Gi", help="Memory limit for the second client job pod.")

    parser.add_argument("--python-path", default="/usr/local/bin/python3", help="Python executable in the job images.")
    parser.add_argument("--ephemeral-storage", default="2Gi", help="Workspace storage for each Kubernetes job pod.")
    parser.add_argument(
        "--server-image",
        help=(
            "Job image for a server that also uses a Kubernetes job launcher. "
            "Omit this when the server uses the process launcher."
        ),
    )
    return parser


def k8s_launcher_spec(image: str, cpu: str, memory: str, python_path: str, ephemeral_storage: str) -> dict:
    return {
        "image": image,
        "python_path": python_path,
        "cpu": cpu,
        "memory": memory,
        "ephemeral_storage": ephemeral_storage,
    }


def create_recipe(args: argparse.Namespace) -> NumpyFedAvgRecipe:
    if args.site_1_name == args.site_2_name:
        raise ValueError("--site-1-name and --site-2-name must identify different clients")
    if "default" in (args.site_1_name, args.site_2_name):
        raise ValueError("client name 'default' is reserved for shared launcher settings")
    if args.num_rounds < 1:
        raise ValueError("--num-rounds must be greater than 0")

    client_sites = (args.site_1_name, args.site_2_name)
    recipe = NumpyFedAvgRecipe(
        name=args.job_name,
        model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        min_clients=len(client_sites),
        num_rounds=args.num_rounds,
        train_script="client.py",
        # Explicit per-site entries target these two clients. Recipe metadata
        # describes resources and launchers; it does not select deploy targets.
        per_site_config={site_name: {} for site_name in client_sites},
        key_metric="weight_mean",
    )

    # Scheduler-facing resource requirements stay separate from Kubernetes
    # container settings. K8sJobLauncher maps num_of_gpus to nvidia.com/gpu.
    set_recipe_meta(
        recipe,
        JobMetaKey.RESOURCE_SPEC,
        {
            args.site_1_name: {"num_of_gpus": args.site_1_gpus},
            args.site_2_name: {"num_of_gpus": args.site_2_gpus},
        },
    )

    launcher_spec = {
        args.site_1_name: {
            "k8s": k8s_launcher_spec(
                args.site_1_image,
                args.site_1_cpu,
                args.site_1_memory,
                args.python_path,
                args.ephemeral_storage,
            )
        },
        args.site_2_name: {
            "k8s": k8s_launcher_spec(
                args.site_2_image,
                args.site_2_cpu,
                args.site_2_memory,
                args.python_path,
                args.ephemeral_storage,
            )
        },
    }
    if args.server_image:
        # The server's provisioned identity may not literally be "server".
        # A default supplies its K8s image while the client blocks above keep
        # their cluster-specific images and settings.
        launcher_spec["default"] = {
            "k8s": k8s_launcher_spec(
                args.server_image,
                "1",
                "2Gi",
                args.python_path,
                args.ephemeral_storage,
            )
        }

    set_recipe_meta(recipe, JobMetaKey.JOB_LAUNCHER_SPEC, launcher_spec)
    return recipe


def main() -> None:
    args = define_parser().parse_args()
    recipe = create_recipe(args)
    env = ProdEnv(
        startup_kit_location=args.startup_kit,
        username=args.username,
        study=args.study,
    )

    run = recipe.execute(env)
    print(f"Job ID: {run.get_job_id()}")
    result = run.get_result()
    print(f"Job status: {run.get_status()}")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()

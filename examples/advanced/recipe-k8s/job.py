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
from pathlib import Path

from model import Cifar10Net

from nvflare.apis.job_def import JobMetaKey
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import ProdEnv, set_recipe_meta


def non_negative_int(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid integer") from None
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed_value


def positive_int(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid integer") from None
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed_value


def gpu_count(value: str) -> int:
    parsed_value = non_negative_int(value)
    if parsed_value > 1:
        raise argparse.ArgumentTypeError("must be 0 or 1 because this example uses one device per client")
    return parsed_value


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit a CIFAR-10 FedAvg recipe to two NVFlare clients that use Kubernetes job launchers.",
        epilog=(
            "Standard Recipe flags --export and --export-dir DIR are consumed by nvflare.recipe "
            "before this parser processes the example-specific options."
        ),
    )
    parser.add_argument("--startup-kit", required=True, help="Path to the production admin startup directory.")
    parser.add_argument("--username", default="admin@nvidia.com", help="Provisioned admin participant name.")
    parser.add_argument("--study", default="default", help="Study in which to submit the job.")
    parser.add_argument("--job-name", default="cifar10-k8s")
    parser.add_argument("--num-rounds", type=positive_int, default=3)

    parser.add_argument("--site-1-name", default="site-1", help="First provisioned NVFlare client name.")
    parser.add_argument("--site-2-name", default="site-2", help="Second provisioned NVFlare client name.")
    parser.add_argument(
        "--image",
        help="Shared client job image. A site-specific image overrides this value for that client.",
    )
    parser.add_argument("--site-1-image", help="Job image pullable by the first client's cluster.")
    parser.add_argument("--site-2-image", help="Job image pullable by the second client's cluster.")
    parser.add_argument("--site-1-gpus", type=gpu_count, default=0)
    parser.add_argument("--site-2-gpus", type=gpu_count, default=0)
    parser.add_argument("--site-1-cpu", default="1", help="Kubernetes CPU limit for the first client job pod.")
    parser.add_argument("--site-2-cpu", default="1", help="Kubernetes CPU limit for the second client job pod.")
    parser.add_argument("--site-1-memory", default="2Gi", help="Memory limit for the first client job pod.")
    parser.add_argument("--site-2-memory", default="2Gi", help="Memory limit for the second client job pod.")
    parser.add_argument("--local-epochs", type=positive_int, default=1, help="Local CIFAR-10 epochs per FL round.")
    parser.add_argument("--batch-size", type=positive_int, default=64, help="Client training batch size.")
    parser.add_argument("--num-workers", type=non_negative_int, default=0, help="Client data-loader workers.")
    parser.add_argument("--data-dir", default="/tmp/nvflare/cifar10", help="CIFAR-10 directory in client job pods.")
    parser.add_argument(
        "--download-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download CIFAR-10 in each client pod; use --no-download-data for a pre-populated data directory.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=non_negative_int,
        default=5000,
        help="Maximum training samples per client; 0 uses the full partition.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=non_negative_int,
        default=1000,
        help="Maximum test samples per client; 0 uses the full partition.",
    )

    parser.add_argument("--python-path", default="/usr/local/bin/python3", help="Python executable in the job images.")
    parser.add_argument("--ephemeral-storage", default="2Gi", help="Workspace storage for each Kubernetes job pod.")
    parser.add_argument(
        "--server-image",
        help=(
            "Job image for a server that also uses a Kubernetes job launcher. "
            "Omit this when the server uses the process launcher."
        ),
    )
    parser.add_argument("--server-cpu", default="1", help="Kubernetes CPU limit for the server job pod.")
    parser.add_argument("--server-memory", default="2Gi", help="Memory limit for the server job pod.")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.site_1_name == args.site_2_name:
        parser.error("--site-1-name and --site-2-name must identify different clients")
    if "default" in (args.site_1_name, args.site_2_name):
        parser.error("client name 'default' is reserved for shared launcher settings")
    if any(character.isspace() for character in args.data_dir):
        parser.error("--data-dir must not contain whitespace")

    missing_image_args = []
    if not (args.site_1_image or args.image):
        missing_image_args.append("--site-1-image")
    if not (args.site_2_image or args.image):
        missing_image_args.append("--site-2-image")
    if missing_image_args:
        parser.error(f"provide --image or specify {', '.join(missing_image_args)}")


def k8s_launcher_spec(image: str, cpu: str, memory: str, python_path: str, ephemeral_storage: str) -> dict:
    return {
        "image": image,
        "python_path": python_path,
        "cpu": cpu,
        "memory": memory,
        "ephemeral_storage": ephemeral_storage,
    }


def client_train_args(args: argparse.Namespace, site_index: int, gpus: int) -> str:
    train_args = [
        "--site-index",
        str(site_index),
        "--num-sites",
        "2",
        "--local-epochs",
        str(args.local_epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--data-dir",
        args.data_dir,
        "--max-train-samples",
        str(args.max_train_samples),
        "--max-test-samples",
        str(args.max_test_samples),
    ]
    if args.download_data:
        train_args.append("--download")
    if gpus:
        train_args.append("--require-gpu")
    return " ".join(train_args)


def create_recipe(args: argparse.Namespace) -> FedAvgRecipe:
    client_sites = (args.site_1_name, args.site_2_name)
    recipe = FedAvgRecipe(
        name=args.job_name,
        model=Cifar10Net(),
        min_clients=len(client_sites),
        num_rounds=args.num_rounds,
        train_script="client.py",
        # Explicit per-site entries target these two clients. Recipe metadata
        # describes resources and launchers; it does not select deploy targets.
        per_site_config={
            args.site_1_name: {"train_args": client_train_args(args, site_index=0, gpus=args.site_1_gpus)},
            args.site_2_name: {"train_args": client_train_args(args, site_index=1, gpus=args.site_2_gpus)},
        },
        key_metric="accuracy",
    )

    # The client training script and the server-side PyTorch persistor both
    # import the model definition, so bundle it into every generated app.
    model_path = str(Path(__file__).with_name("model.py"))
    recipe.job.add_file_to_server(model_path)
    for site_name in client_sites:
        recipe.job.add_file_to(model_path, site_name)

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
                args.site_1_image or args.image,
                args.site_1_cpu,
                args.site_1_memory,
                args.python_path,
                args.ephemeral_storage,
            )
        },
        args.site_2_name: {
            "k8s": k8s_launcher_spec(
                args.site_2_image or args.image,
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
                args.server_cpu,
                args.server_memory,
                args.python_path,
                args.ephemeral_storage,
            )
        }

    set_recipe_meta(recipe, JobMetaKey.JOB_LAUNCHER_SPEC, launcher_spec)
    return recipe


def main() -> None:
    parser = define_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    recipe = create_recipe(args)
    env = ProdEnv(
        startup_kit_location=args.startup_kit,
        username=args.username,
        study=args.study,
    )

    run = recipe.execute(env)
    print(f"Job ID: {run.get_job_id()}", flush=True)
    print("Waiting for job to complete...", flush=True)
    result = run.get_result()
    print(f"Job status: {run.get_status()}")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()

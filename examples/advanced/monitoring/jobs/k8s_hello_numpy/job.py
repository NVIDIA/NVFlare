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
"""Submit a minimal monitored NumPy job to a production NVFlare environment."""

import argparse

from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.client.config import TransferType
from nvflare.fuel_opt.statsd.statsd_reporter import StatsDReporter
from nvflare.metrics.job_metrics_collector import JobMetricsCollector
from nvflare.recipe.prod_env import ProdEnv


def _parse_client_sites(raw: str) -> list[str]:
    sites = [site.strip() for site in raw.split(",") if site.strip()]
    if not sites:
        raise ValueError("at least one client site must be provided")
    return sites


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit a monitored hello-numpy job to a production NVFlare system.")
    parser.add_argument("--startup_kit_location", required=True, help="Path to the admin startup kit directory.")
    parser.add_argument("--username", default="admin@nvidia.com", help="Username for production authentication.")
    parser.add_argument("--login_timeout", type=float, default=10.0, help="Session login timeout in seconds.")
    parser.add_argument("--statsd_host", default="statsd-exporter.nvflare-monitoring.svc.cluster.local")
    parser.add_argument("--statsd_port", type=int, default=9125)
    parser.add_argument("--client_sites", default="site-2", help="Comma-separated client site names.")
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--update_type", choices=["full", "diff"], default="full")
    return parser


def main():
    args = _build_parser().parse_args()
    client_sites = _parse_client_sites(args.client_sites)
    client_tags = {"env": "k8s"}
    if len(client_sites) == 1:
        client_tags["site"] = client_sites[0]

    recipe = NumpyFedAvgRecipe(
        name="hello-numpy-k8s-monitoring",
        min_clients=len(client_sites),
        num_rounds=args.num_rounds,
        model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        train_script="client.py",
        train_args=f"--update_type {args.update_type}",
        params_transfer_type=TransferType.FULL if args.update_type == "full" else TransferType.DIFF,
    )

    recipe.job.to_server(
        JobMetricsCollector(tags={"site": "server", "env": "k8s"}, streaming_to_server=False),
        id="server_job_metrics_collector",
    )
    recipe.job.to_server(
        StatsDReporter(site="server", host=args.statsd_host, port=args.statsd_port),
        id="server_statsd_reporter",
    )
    recipe.job.to_clients(
        JobMetricsCollector(tags=client_tags),
        id="client_job_metrics_collector",
    )
    recipe.job.to_clients(
        StatsDReporter(
            site=client_sites[0] if len(client_sites) == 1 else "",
            host=args.statsd_host,
            port=args.statsd_port,
        ),
        id="client_statsd_reporter",
    )

    env = ProdEnv(
        startup_kit_location=args.startup_kit_location,
        username=args.username,
        login_timeout=args.login_timeout,
    )
    run = recipe.execute(env)
    print()
    result_dir = run.get_result(timeout=0.0)
    if result_dir is None:
        raise RuntimeError("Job did not complete successfully or result is unavailable.")
    print("Result can be found in:", result_dir)
    print("Job Status is:", run.get_status())


if __name__ == "__main__":
    main()

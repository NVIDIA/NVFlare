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
import os
import tempfile

from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.client.config import TransferType
from nvflare.fuel.flare_api.api_spec import MonitorReturnCode
from nvflare.fuel.flare_api.flare_api import new_secure_session
from nvflare.fuel_opt.statsd.statsd_reporter import StatsDReporter
from nvflare.metrics.job_metrics_collector import JobMetricsCollector


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


def _monitor_and_download_result(session, job_id: str) -> str:
    first = {"value": True}

    def _monitor_cb(sess, monitored_job_id, job_meta, *cb_args, **cb_kwargs):
        if first["value"]:
            print("Job ID: ", monitored_job_id)
            print("Job Meta: ", job_meta)
            first["value"] = False
        elif job_meta["status"] == "RUNNING":
            print(".", end="")
        else:
            print("\n" + str(job_meta))
        return True

    rc = session.monitor_job(job_id, timeout=0.0, cb=_monitor_cb)
    print(f"\njob monitor done: rc={rc!r}")
    if rc != MonitorReturnCode.JOB_FINISHED:
        raise RuntimeError(f"job {job_id} did not finish cleanly: {rc}")
    return session.download_job_result(job_id)


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

    with tempfile.TemporaryDirectory() as temp_dir:
        recipe.job.export_job(temp_dir)
        job_dir = os.path.join(temp_dir, recipe.job.name)

        session = new_secure_session(
            username=args.username,
            startup_kit_location=args.startup_kit_location,
            timeout=args.login_timeout,
        )
        try:
            print("Connecting to FLARE ...")
            job_id = session.submit_job(job_dir)
            print(f"Submitted job '{recipe.job.name}' with ID: {job_id}")
            print()
            print("Connecting to FLARE ...")
            result_dir = _monitor_and_download_result(session, job_id)
            print()
            print("Result can be found in:", result_dir)
            print("Job Status is:", session.get_job_status(job_id))
            print()
        finally:
            session.close()


if __name__ == "__main__":
    main()

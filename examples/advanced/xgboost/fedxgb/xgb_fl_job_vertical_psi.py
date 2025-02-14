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

import argparse

from src.local_psi import LocalPSI

from nvflare.app_common.psi.dh_psi.dh_psi_controller import DhPSIController
from nvflare.app_common.psi.file_psi_writer import FilePSIWriter
from nvflare.app_common.psi.psi_executor import PSIExecutor
from nvflare.app_opt.psi.dh_psi.dh_psi_task_handler import DhPSITaskHandler
from nvflare.job_config.api import FedJob


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_split_path",
        type=str,
        default="/tmp/nvflare/dataset/xgboost_higgs_vertical/site-x/higgs.data.csv",
        help="Path to data split files for each site",
    )
    parser.add_argument("--site_num", type=int, default=2, help="Total number of sites")
    parser.add_argument("--psi_path", type=str, default="psi/intersection.txt", help="PSI ouput path")
    return parser.parse_args()


def main():
    args = define_parser()
    data_split_path = args.data_split_path
    psi_path = args.psi_path
    site_num = args.site_num
    job_name = "xgboost_vertical_psi"
    job = FedJob(name=job_name, min_clients=site_num)

    # Define the controller workflow and send to server
    controller = DhPSIController()
    job.to_server(controller)

    # Define the executor and other components for each site
    executor = PSIExecutor(psi_algo_id="dh_psi")
    job.to_clients(executor, id="psi_executor", tasks=["PSI"])

    local_psi = LocalPSI(psi_writer_id="psi_writer", data_split_path=data_split_path, id_col="uid")
    job.to_clients(local_psi, id="local_psi")

    task_handler = DhPSITaskHandler(local_psi_id="local_psi")
    job.to_clients(task_handler, id="dh_psi")

    psi_writer = FilePSIWriter(output_path=psi_path)
    job.to_clients(psi_writer, id="psi_writer")

    # Export job config and run the job
    job.export_job("/tmp/nvflare/workspace/jobs/")
    job.simulator_run(f"/tmp/nvflare/workspace/works/{job_name}", n_clients=site_num)


if __name__ == "__main__":
    main()

# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

"""Federated Simulator Multi-GPU client process launching script."""

import argparse
import os
import sys

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.fuel.hci.server.authz import AuthorizationService
from nvflare.fuel.sec.audit import AuditService
from nvflare.private.defs import AppFolderConstants
from nvflare.private.fed.app.deployer.simulator_deployer import SimulatorDeploy
from nvflare.private.fed.app.simulator.simulator_runner import SimulatorClientRunner
from nvflare.security.security import EmptyAuthorizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-o", type=str, help="WORKSPACE folder", required=True)
    parser.add_argument("--client_names", "-c", type=str, help="client names", required=True)
    parser.add_argument("--gpu", "-g", type=str, help="gpu index number", required=True)
    parser.add_argument("--server_ports", "-p", type=str, help="server port", required=True)
    parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")
    args = parser.parse_args()

    # self.logger = logging.getLogger()
    args.threads = 1
    args.log_config = None
    args.config_folder = "config"
    args.job_id = "simulate_job"
    args.client_config = os.path.join(args.config_folder, "config_fed_client.json")
    args.env = os.path.join("config", AppFolderConstants.CONFIG_ENV)

    os.chdir(args.workspace)
    AuthorizationService.initialize(EmptyAuthorizer())
    AuditService.initialize(audit_file_name=WorkspaceConstants.AUDIT_LOG)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    client_names = args.client_names.split(",")
    server_ports = []
    for part in args.server_ports.split(","):
        server_ports.append(int(part))
    deployer = SimulatorDeploy(server_ports)
    client_runner = SimulatorClientRunner(args, client_names, deployer)
    client_runner.run()


if __name__ == "__main__":
    """
    This is the main program of client run process when running the NVFlare Simulator using multi-GPU.
    """

    if sys.version_info >= (3, 9):
        raise RuntimeError("Python versions 3.9 and above are not yet supported. Please use Python 3.8 or 3.7.")
    if sys.version_info < (3, 7):
        raise RuntimeError("Python versions 3.6 and below are not supported. Please use Python 3.8 or 3.7.")

    main()

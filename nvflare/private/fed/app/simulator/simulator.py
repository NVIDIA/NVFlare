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

"""Federated server launching script."""

import argparse
import logging
import logging.config
import os
import shutil
import sys

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.hci.server.authz import AuthorizationService
from nvflare.fuel.sec.audit import AuditService
from nvflare.private.defs import AppFolderConstants
from nvflare.private.fed.app.deployer.simulator_deployer import SimulatorDeploy
from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner
from nvflare.security.security import EmptyAuthorizer


def main():
    if sys.version_info >= (3, 9):
        raise RuntimeError("Python versions 3.9 and above are not yet supported. Please use Python 3.8 or 3.7.")
    if sys.version_info < (3, 7):
        raise RuntimeError("Python versions 3.6 and below are not supported. Please use Python 3.8 or 3.7.")
    parser = argparse.ArgumentParser()
    parser.add_argument("job_folder")
    parser.add_argument("--workspace", "-m", type=str, help="WORKSPACE folder", required=True)
    parser.add_argument("--clients", "-n", type=int, help="number of clients", required=True)
    parser.add_argument("--threads", "-t", type=int, help="number of running threads", required=True)

    parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")

    args = parser.parse_args()

    log_config_file_path = os.path.join(args.workspace, "startup", "log.config")
    assert os.path.isfile(log_config_file_path), "missing log config file {}".format(log_config_file_path)
    logging.config.fileConfig(fname=log_config_file_path, disable_existing_loggers=False)

    logger = logging.getLogger()
    args.log_config = None
    args.config_folder = "config"
    args.job_id = "simulate_job"
    args.client_config = os.path.join(args.config_folder, "config_fed_client.json")
    args.env = os.path.join("config", AppFolderConstants.CONFIG_ENV)

    try:
        os.chdir(args.workspace)
        AuthorizationService.initialize(EmptyAuthorizer())
        AuditService.initialize(audit_file_name=WorkspaceConstants.AUDIT_LOG)

        simulator_root = os.path.join(args.workspace, "simulate_job")
        if os.path.exists(simulator_root):
            shutil.rmtree(simulator_root)

        deployer = SimulatorDeploy()

        try:
            # Deploy the FL server
            simulator_server, services = deployer.create_fl_server(args)
            services.deploy(args, grpc_args=simulator_server)

            # Deploy the FL client
            client_name = "client1"
            federated_client = deployer.create_fl_client(client_name, args)

            simulator_runner = SimulatorRunner()
            simulator_runner.run(simulator_root, args, logger, services, federated_client)

        finally:
            deployer.close()

        logger.info("Server started")

    except ConfigError as ex:
        print("ConfigError:", str(ex))
    finally:
        pass


if __name__ == "__main__":
    """
    This is the main program when starting the NVIDIA FLARE server process.
    """

    main()

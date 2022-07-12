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
import os
import shutil
import sys

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.fuel.sec.audit import AuditService
from nvflare.fuel.common.excepts import ConfigError
from nvflare.private.defs import AppFolderConstants
from nvflare.private.fed.app.deployer.simulator_deployer import SimulatorDeploy
from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner
from nvflare.private.fed.client.admin import FedAdminAgent
from nvflare.private.fed.client.admin_msg_sender import AdminMessageSender
from nvflare.private.fed.client.client_engine import ClientEngine
from nvflare.private.fed.client.fed_client import FederatedClient


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

    # config_folder = kv_list.get("config_folder", "")
    # if config_folder == "":
    #     args.server_config = AppFolderConstants.CONFIG_FED_SERVER
    # else:
    #     args.server_config = os.path.join(config_folder, AppFolderConstants.CONFIG_FED_SERVER)

    # # TODO:: remove env and train config since they are not core
    # args.env = os.path.join("config", AppFolderConstants.CONFIG_ENV)
    # args.train_config = os.path.join("config", AppFolderConstants.CONFIG_TRAIN)
    # args.config_folder = config_folder
    logger = logging.getLogger()
    args.log_config = None
    args.config_folder = "config"
    args.env = os.path.join("config", AppFolderConstants.CONFIG_ENV)

    # for name in [WorkspaceConstants.RESTART_FILE, WorkspaceConstants.SHUTDOWN_FILE]:
    #     try:
    #         f = os.path.join(args.workspace, name)
    #         if os.path.exists(f):
    #             os.remove(f)
    #     except BaseException:
    #         print("Could not remove the {} file.  Please check your system before starting FL.".format(name))
    #         sys.exit(-1)

    try:
        os.chdir(args.workspace)
        AuditService.initialize(audit_file_name=WorkspaceConstants.AUDIT_LOG)

        simulator_root = os.path.join(args.workspace, "simulate_job")
        if os.path.exists(simulator_root):
            shutil.rmtree(simulator_root)

        # startup = os.path.join(args.workspace, "startup")
        # conf = FLServerStarterConfiger(
        #     app_root=startup,
        #     server_config_file_name=args.fed_server,
        #     log_config_file_name=WorkspaceConstants.LOGGING_CONFIG,
        #     kv_list=args.set,
        # )
        # log_level = os.environ.get("FL_LOG_LEVEL", "")
        # numeric_level = getattr(logging, log_level.upper(), None)
        # if isinstance(numeric_level, int):
        #     logging.getLogger().setLevel(numeric_level)
        #     logger.debug("loglevel debug enabled")
        #     logger.info("loglevel info enabled")
        #     logger.warning("loglevel warn enabled")
        #     logger.error("loglevel error enabled")
        #     logger.critical("loglevel critical enabled")
        # conf.configure()

        # log_file = os.path.join(args.workspace, "log.txt")
        # add_logfile_handler(log_file)

        deployer = SimulatorDeploy()
        # secure_train = conf.cmd_vars.get("secure_train", False)

        # security_check(secure_train=secure_train, content_folder=startup, fed_server_config=args.fed_server)

        try:
            # Deploy the FL server
            # services = deployer.deploy(args)
            simulator_server, services = deployer.create_fl_server(args)
            services.deploy(args, grpc_args=simulator_server)

            federated_client = deployer.create_fl_client(args)
            # federated_client.register()
            # federated_client.start_heartbeat()
            # servers = [{t["name"]: t["service"]} for t in deployer.server_config]
            # admin_agent = create_admin_agent(
            #     sorted(servers)[0],
            #     federated_client,
            #     args,
            # )
            # admin_agent.start()

            simulator_runner = SimulatorRunner()

            simulator_runner.run(simulator_root, args, logger, services)

            # first_server = sorted(conf.config_data["servers"])[0]
            # allow command to overwrite the admin_host
            # if conf.cmd_vars.get("host", None):
            #     first_server["admin_host"] = conf.cmd_vars["host"]
            # admin_server = create_admin_server(
            #     services,
            #     server_conf=first_server,
            #     args=args,
            #     secure_train=False,
            #     app_validator=deployer.app_validator,
            # )
            # admin_server.start()

            # services.platform = "PT"

            # services.set_admin_server(admin_server)
        finally:
            deployer.close()

        logger.info("Server started")

    except ConfigError as ex:
        print("ConfigError:", str(ex))
    finally:
        pass


def create_admin_agent(
    server_args,
    federated_client: FederatedClient,
    args,
    rank=0
):
    sender = AdminMessageSender(
        client_name=federated_client.token,
        server_args=server_args,
        secure=False,
    )
    client_engine = ClientEngine(federated_client, federated_client.token, sender, args, rank)
    admin_agent = FedAdminAgent(
        client_name="admin_agent",
        sender=sender,
        app_ctx=client_engine,
    )
    admin_agent.app_ctx.set_agent(admin_agent)
    federated_client.set_client_engine(client_engine)

    client_engine.fire_event(EventType.SYSTEM_START, client_engine.new_context())

    return admin_agent


if __name__ == "__main__":
    """
    This is the main program when starting the NVIDIA FLARE server process.
    """

    main()

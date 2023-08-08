# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""Federated client launching script."""

import argparse
import os
import sys
import time

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import JobConstants, SiteType, WorkspaceConstants
from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.f3.mpm import MainProcessMonitor as mpm
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.private.defs import AppFolderConstants
from nvflare.private.fed.app.fl_conf import FLClientStarterConfiger, create_privacy_manager
from nvflare.private.fed.client.admin import FedAdminAgent
from nvflare.private.fed.client.client_engine import ClientEngine
from nvflare.private.fed.client.client_status import ClientStatus
from nvflare.private.fed.client.fed_client import FederatedClient
from nvflare.private.fed.utils.fed_utils import add_logfile_handler, fobs_initialize, security_init
from nvflare.private.privacy_manager import PrivacyService
from nvflare.security.logging import secure_format_exception


def main():
    if sys.version_info < (3, 7):
        raise RuntimeError("Please use Python 3.7 or above.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-m", type=str, help="WORKSPACE folder", required=True)
    parser.add_argument("--fed_client", "-s", type=str, help="client config json file", required=True)
    parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    kv_list = parse_vars(args.set)

    config_folder = kv_list.get("config_folder", "")
    if config_folder == "":
        args.client_config = JobConstants.CLIENT_JOB_CONFIG
    else:
        args.client_config = os.path.join(config_folder, JobConstants.CLIENT_JOB_CONFIG)
    # TODO:: remove env and train config since they are not core
    args.env = os.path.join("config", AppFolderConstants.CONFIG_ENV)
    args.train_config = os.path.join("config", AppFolderConstants.CONFIG_TRAIN)
    args.log_config = None
    args.job_id = None

    workspace = Workspace(root_dir=args.workspace)

    for name in [WorkspaceConstants.RESTART_FILE, WorkspaceConstants.SHUTDOWN_FILE]:
        try:
            f = workspace.get_file_path_in_root(name)
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            print("Could not remove the {} file.  Please check your system before starting FL.".format(name))
            sys.exit(-1)

    rank = args.local_rank

    try:
        os.chdir(args.workspace)
        fobs_initialize()

        conf = FLClientStarterConfiger(
            workspace=workspace,
            args=args,
            kv_list=args.set,
        )
        conf.configure()

        log_file = workspace.get_log_file_path()
        add_logfile_handler(log_file)

        deployer = conf.base_deployer
        security_init(
            secure_train=deployer.secure_train,
            site_org=conf.site_org,
            workspace=workspace,
            app_validator=conf.app_validator,
            site_type=SiteType.CLIENT,
        )

        # initialize Privacy Service
        privacy_manager = create_privacy_manager(workspace, names_only=True)
        PrivacyService.initialize(privacy_manager)

        federated_client = deployer.create_fed_client(args)
        federated_client.start_overseer_agent()

        while not federated_client.sp_established:
            print("Waiting for SP....")
            time.sleep(1.0)

        federated_client.use_gpu = False
        federated_client.config_folder = config_folder

        while federated_client.cell is None:
            print("Waiting client cell to be created ....")
            time.sleep(1.0)

        federated_client.register()

        if not federated_client.token:
            print("The client could not register to server. ")
            raise RuntimeError("Login failed.")

        federated_client.start_heartbeat(interval=kv_list.get("heart_beat_interval", 10.0))

        admin_agent = create_admin_agent(
            deployer.req_processors,
            federated_client,
            args,
            rank,
        )

        while federated_client.status != ClientStatus.STOPPED:
            time.sleep(1.0)

        deployer.close()

    except ConfigError as e:
        print(f"ConfigError: {secure_format_exception(e)}")


def create_admin_agent(
    req_processors,
    federated_client: FederatedClient,
    args,
    rank,
):
    """Creates an admin agent.

    Args:
        req_processors: request processors
        federated_client: FL client object
        args: command args
        rank: client rank process number

    Returns:
        A FedAdminAgent.
    """
    client_engine = ClientEngine(federated_client, federated_client.token, args, rank)
    admin_agent = FedAdminAgent(
        client_name="admin_agent",
        cell=federated_client.cell,
        app_ctx=client_engine,
    )
    client_engine.set_agent(admin_agent)
    federated_client.set_client_engine(client_engine)
    for processor in req_processors:
        admin_agent.register_processor(processor)

    client_engine.fire_event(EventType.SYSTEM_START, client_engine.new_context())

    return admin_agent


if __name__ == "__main__":
    """
    This is the main program when starting the NVIDIA FLARE client process.
    """
    # # For MacOS, it needs to use 'spawn' for creating multi-process.
    # if os.name == 'posix':
    #     import multiprocessing
    #     multiprocessing.set_start_method('spawn')

    # import multiprocessing
    # multiprocessing.set_start_method('spawn')

    # main()
    rc = mpm.run(main_func=main)
    sys.exit(rc)

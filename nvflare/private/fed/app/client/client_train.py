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

"""Federated client launching script."""

import argparse
import os
import sys
import time

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.sec.audit import AuditService
from nvflare.fuel.sec.security_content_service import SecurityContentService
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.private.defs import AppFolderConstants, SSLConstants
from nvflare.private.fed.app.fl_conf import FLClientStarterConfiger
from nvflare.private.fed.client.admin import FedAdminAgent
from nvflare.private.fed.client.admin_msg_sender import AdminMessageSender
from nvflare.private.fed.client.client_engine import ClientEngine
from nvflare.private.fed.client.fed_client import FederatedClient
from nvflare.private.fed.utils.fed_utils import add_logfile_handler, secure_content_check


def main():
    if sys.version_info >= (3, 9):
        raise RuntimeError("Python versions 3.9 and above are not yet supported. Please use Python 3.8 or 3.7.")
    if sys.version_info < (3, 7):
        raise RuntimeError("Python versions 3.6 and below are not supported. Please use Python 3.8 or 3.7.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-m", type=str, help="WORKSPACE folder", required=True)
    parser.add_argument(
        "--fed_client", "-s", type=str, help="an aggregation server specification json file", required=True
    )
    parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    kv_list = parse_vars(args.set)

    config_folder = kv_list.get("config_folder", "")
    if config_folder == "":
        args.client_config = AppFolderConstants.CONFIG_FED_CLIENT
    else:
        args.client_config = os.path.join(config_folder, AppFolderConstants.CONFIG_FED_CLIENT)
    # TODO:: remove env and train config since they are not core
    args.env = os.path.join("config", AppFolderConstants.CONFIG_ENV)
    args.train_config = os.path.join("config", AppFolderConstants.CONFIG_TRAIN)
    args.log_config = None

    for name in [WorkspaceConstants.RESTART_FILE, WorkspaceConstants.SHUTDOWN_FILE]:
        try:
            f = os.path.join(args.workspace, name)
            if os.path.exists(f):
                os.remove(f)
        except BaseException:
            print("Could not remove the {} file.  Please check your system before starting FL.".format(name))
            sys.exit(-1)

    rank = args.local_rank

    try:
        os.chdir(args.workspace)
        AuditService.initialize(audit_file_name=WorkspaceConstants.AUDIT_LOG)

        startup = os.path.join(args.workspace, "startup")
        conf = FLClientStarterConfiger(
            app_root=startup,
            client_config_file_name=args.fed_client,
            log_config_file_name=WorkspaceConstants.LOGGING_CONFIG,
            kv_list=args.set,
        )
        conf.configure()

        log_file = os.path.join(args.workspace, "log.txt")
        add_logfile_handler(log_file)

        deployer = conf.base_deployer

        security_check(secure_train=deployer.secure_train, content_folder=startup, fed_client_config=args.fed_client)

        federated_client = deployer.create_fed_client(args)

        while not federated_client.sp_established:
            print("Waiting for SP....")
            time.sleep(1.0)

        federated_client.use_gpu = False
        federated_client.config_folder = config_folder

        if rank == 0:
            federated_client.register()

        if not federated_client.token:
            print("The client could not register to server. ")
            raise RuntimeError("Login failed.")

        federated_client.start_heartbeat()

        servers = [{t["name"]: t["service"]} for t in deployer.server_config]
        admin_agent = create_admin_agent(
            deployer.client_config,
            deployer.client_name,
            deployer.req_processors,
            deployer.secure_train,
            sorted(servers)[0],
            federated_client,
            args,
            deployer.multi_gpu,
            rank,
        )
        admin_agent.start()

        deployer.close()

    except ConfigError as ex:
        print("ConfigError:", str(ex))
    finally:
        pass

    sys.exit(0)


def security_check(secure_train: bool, content_folder: str, fed_client_config: str):
    """To check the security content if running in security mode.

    Args:
       secure_train (bool): if run in secure mode or not.
       content_folder (str): the folder to check.
       fed_client_config (str): fed_client.json
    """
    # initialize the SecurityContentService.
    # must do this before initializing other services since it may be needed by them!
    SecurityContentService.initialize(content_folder=content_folder)

    if secure_train:
        insecure_list = secure_content_check(fed_client_config, site_type="client")
        if len(insecure_list):
            print("The following files are not secure content.")
            for item in insecure_list:
                print(item)
            sys.exit(1)
    # initialize the AuditService, which is used by command processing.
    # The Audit Service can be used in other places as well.
    AuditService.initialize(audit_file_name=WorkspaceConstants.AUDIT_LOG)


def create_admin_agent(
    client_args,
    client_id,
    req_processors,
    secure_train,
    server_args,
    federated_client: FederatedClient,
    args,
    is_multi_gpu,
    rank,
):
    """Creates an admin agent.

    Args:
        client_args: start client command args
        client_id: client name
        req_processors: request processors
        secure_train: True/False
        server_args: FL server args
        federated_client: FL client object
        args: command args
        is_multi_gpu: True/False
        rank: client rank process number

    Returns:
        A FedAdminAgent.
    """
    sender = AdminMessageSender(
        client_name=federated_client.token,
        root_cert=client_args[SSLConstants.ROOT_CERT],
        ssl_cert=client_args[SSLConstants.CERT],
        private_key=client_args[SSLConstants.PRIVATE_KEY],
        server_args=server_args,
        secure=secure_train,
        is_multi_gpu=is_multi_gpu,
        rank=rank,
    )
    client_engine = ClientEngine(federated_client, federated_client.token, sender, args, rank)
    admin_agent = FedAdminAgent(
        client_name="admin_agent",
        sender=sender,
        app_ctx=client_engine,
    )
    admin_agent.app_ctx.set_agent(admin_agent)
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

    main()

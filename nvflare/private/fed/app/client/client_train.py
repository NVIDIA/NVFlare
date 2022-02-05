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

"""Provides a command line interface for a federated client."""

import argparse
import os
import sys

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.sec.audit import AuditService
from nvflare.fuel.sec.security_content_service import LoadResult, SecurityContentService
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.private.fed.app.fl_conf import FLClientStarterConfiger
from nvflare.private.fed.client.admin import FedAdminAgent
from nvflare.private.fed.client.admin_msg_sender import AdminMessageSender
from nvflare.private.fed.client.client_engine import ClientEngine


def main():
    """Start program of the FL client."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-m", type=str, help="WORKSPACE folder", required=True)

    parser.add_argument(
        "--fed_client", "-s", type=str, help="an aggregation server specification json file", required=True
    )

    parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")

    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    kv_list = parse_vars(args.set)

    args.train_config = "config/config_train.json"
    config_folder = kv_list.get("config_folder", "")
    if config_folder == "":
        args.client_config = "config_fed_client.json"
    else:
        args.client_config = config_folder + "/config_fed_client.json"
    args.env = "config/environment.json"
    args.log_config = None

    try:
        remove_restart_file(args)
    except BaseException:
        print("Could not remove the restart.fl / shutdown.fl file.  Please check your system before starting FL.")
        sys.exit(-1)

    rank = args.local_rank

    try:
        os.chdir(args.workspace)
        AuditService.initialize(audit_file_name="audit.log")

        workspace = os.path.join(args.workspace, "startup")

        # trainer = WorkFlowFactory().create_client_trainer(train_configs, envs)
        conf = FLClientStarterConfiger(
            app_root=workspace,
            # wf_config_file_name="config_train.json",
            client_config_file_name=args.fed_client,
            # env_config_file_name="environment.json",
            log_config_file_name="log.config",
            kv_list=args.set,
        )
        conf.configure()

        trainer = conf.base_deployer

        security_check(trainer.secure_train, args)

        federated_client = trainer.create_fed_client()
        # federated_client.platform = conf.wf_config_data.get("platform", "PT")
        federated_client.use_gpu = False
        # federated_client.cross_site_validate = kv_list.get("cross_site_validate", True)
        federated_client.config_folder = config_folder

        if rank == 0:
            federated_client.register()

        if not federated_client.token:
            print("The client could not register to server. ")
            raise RuntimeError("Login failed.")

        federated_client.start_heartbeat()

        servers = [{t["name"]: t["service"]} for t in trainer.server_config]

        admin_agent = create_admin_agent(
            trainer.client_config,
            trainer.client_name,
            trainer.req_processors,
            trainer.secure_train,
            sorted(servers)[0],
            federated_client,
            args,
            trainer.multi_gpu,
            rank,
        )
        admin_agent.start()

        trainer.close()

    except ConfigError as ex:
        print("ConfigError:", str(ex))
    finally:
        # shutil.rmtree(workspace)
        pass

    sys.exit(0)


def security_check(secure_train, args):
    """To check the security content if running in security mode.

    Args:
        secure_train: True/False
        args: command args

    """
    # initialize the SecurityContentService.
    # must do this before initializing other services since it may be needed by them!
    startup = os.path.join(args.workspace, "startup")
    SecurityContentService.initialize(content_folder=startup)

    if secure_train:
        insecure_list = secure_content_check(args)
        if len(insecure_list):
            print("The following files are not secure content.")
            for item in insecure_list:
                print(item)
            sys.exit(1)
    # initialize the AuditService, which is used by command processing.
    # The Audit Service can be used in other places as well.
    AuditService.initialize(audit_file_name="audit.log")


def secure_content_check(args):
    """To check the security contents.

    Args:
        args: command args

    Returns: the insecure content list

    """
    insecure_list = []
    data, sig = SecurityContentService.load_json(args.fed_client)
    if sig != LoadResult.OK:
        insecure_list.append(args.fed_client)

    client = data["client"]
    content, sig = SecurityContentService.load_content(client.get("ssl_cert"))
    if sig != LoadResult.OK:
        insecure_list.append(client.get("ssl_cert"))
    content, sig = SecurityContentService.load_content(client.get("ssl_private_key"))
    if sig != LoadResult.OK:
        insecure_list.append(client.get("ssl_private_key"))
    content, sig = SecurityContentService.load_content(client.get("ssl_root_cert"))
    if sig != LoadResult.OK:
        insecure_list.append(client.get("ssl_root_cert"))

    return insecure_list


def remove_restart_file(args):
    """To remove the restart.fl file.

    Args:
        args: command args

    """
    restart_file = os.path.join(args.workspace, "restart.fl")
    if os.path.exists(restart_file):
        os.remove(restart_file)
    restart_file = os.path.join(args.workspace, "shutdown.fl")
    if os.path.exists(restart_file):
        os.remove(restart_file)


def create_admin_agent(
    client_args, client_id, req_processors, secure_train, server_args, federated_client, args, is_multi_gpu, rank
):
    """To create the admin client.

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

    Returns: admin client

    """
    sender = AdminMessageSender(
        client_name=federated_client.token,
        root_cert=client_args["ssl_root_cert"],
        ssl_cert=client_args["ssl_cert"],
        private_key=client_args["ssl_private_key"],
        server_args=server_args,
        secure=secure_train,
        is_multi_gpu=is_multi_gpu,
        rank=rank,
    )
    admin_agent = FedAdminAgent(
        client_name="admin_agent",
        sender=sender,
        app_ctx=ClientEngine(federated_client, federated_client.token, sender, args, rank),
    )
    admin_agent.app_ctx.set_agent(admin_agent)
    for processor in req_processors:
        admin_agent.register_processor(processor)

    return admin_agent
    # self.admin_agent.start()


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

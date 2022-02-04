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

"""Provides a command line interface for a federated client trainer."""

import argparse
import os
import sys
import traceback

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.workspace import Workspace
from nvflare.fuel.sec.security_content_service import SecurityContentService
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.private.defs import EngineConstant
from nvflare.private.fed.app.fl_conf import FLClientStarterConfiger
from nvflare.private.fed.client.client_json_config import ClientJsonConfigurator
from nvflare.private.fed.client.client_run_manager import ClientRunManager
from nvflare.private.fed.client.client_runner import ClientRunner
from nvflare.private.fed.client.client_status import ClientStatus
from nvflare.private.fed.client.command_agent import CommandAgent


def main():
    """Worker_process start program."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-m", type=str, help="WORKSPACE folder", required=True)
    parser.add_argument("--startup", "-w", type=str, help="startup folder", required=True)

    parser.add_argument(
        "--fed_client", "-s", type=str, help="an aggregation server specification json file", required=True
    )

    parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")

    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    kv_list = parse_vars(args.set)

    args.train_config = os.path.join("config", "config_train.json")
    config_folder = kv_list.get("config_folder", "")
    secure_train = kv_list.get("secure_train", True)
    if config_folder == "":
        args.client_config = "config_fed_client.json"
    else:
        args.client_config = os.path.join(config_folder, "config_fed_client.json")
    args.config_folder = config_folder
    args.env = os.path.join("config", "environment.json")

    try:
        remove_restart_file(args)
    except BaseException:
        print("Could not remove the restart.fl / shutdown.fl file.  Please check your system before starting FL.")
        sys.exit(-1)

    restart_file = os.path.join(args.workspace, "restart.fl")
    if os.path.exists(restart_file):
        os.remove(restart_file)

    print("starting the client .....")

    deployer = None
    command_agent = None

    startup = os.path.join(args.workspace, "startup")
    SecurityContentService.initialize(content_folder=startup)

    try:
        token_file = os.path.join(args.workspace, EngineConstant.CLIENT_TOKEN_FILE)
        with open(token_file, "r") as f:
            token = f.readline().strip()
            run_number = f.readline().strip()
            client_name = f.readline().strip()
            listen_port = f.readline().strip()
            print(
                "token is: {} run_number is: {} client_name: {} listen_port: {}".format(
                    token, run_number, client_name, listen_port
                )
            )

        startup = args.startup
        app_root = os.path.join(args.workspace, "run_" + str(run_number), "app_" + client_name)

        app_log_config = os.path.join(app_root, config_folder, "log.config")
        if os.path.exists(app_log_config):
            args.log_config = app_log_config
        else:
            args.log_config = os.path.join(startup, "log.config")

        conf = FLClientStarterConfiger(
            app_root=startup,
            client_config_file_name=args.fed_client,
            log_config_file_name=args.log_config,
            kv_list=args.set,
        )
        conf.configure()

        deployer = conf.base_deployer
        federated_client = deployer.create_fed_client()
        federated_client.status = ClientStatus.STARTING

        federated_client.token = token
        federated_client.client_name = client_name
        federated_client.fl_ctx.set_prop(FLContextKey.CLIENT_NAME, client_name, private=False)
        federated_client.fl_ctx.set_prop(EngineConstant.FL_TOKEN, token, private=False)
        federated_client.fl_ctx.set_prop(FLContextKey.WORKSPACE_ROOT, args.workspace, private=True)

        client_config_file_name = os.path.join(app_root, args.client_config)
        conf = ClientJsonConfigurator(
            config_file_name=client_config_file_name,
        )
        conf.configure()

        workspace = Workspace(args.workspace, client_name, config_folder)
        run_manager = ClientRunManager(
            client_name=client_name,
            run_num=int(run_number),
            workspace=workspace,
            client=federated_client,
            components=conf.runner_config.components,
            handlers=conf.runner_config.handlers,
            conf=conf,
        )
        federated_client.run_manager = run_manager

        with run_manager.new_context() as fl_ctx:
            fl_ctx.set_prop(FLContextKey.CLIENT_NAME, client_name, private=False)
            fl_ctx.set_prop(EngineConstant.FL_TOKEN, token, private=False)
            fl_ctx.set_prop(FLContextKey.WORKSPACE_ROOT, args.workspace, private=True)
            fl_ctx.set_prop(FLContextKey.ARGS, args, sticky=True)
            fl_ctx.set_prop(FLContextKey.APP_ROOT, app_root, private=True, sticky=True)
            fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True)
            fl_ctx.set_prop(FLContextKey.SECURE_MODE, secure_train, private=True, sticky=True)

            client_runner = ClientRunner(config=conf.runner_config, run_num=int(run_number), engine=run_manager)
            run_manager.add_handler(client_runner)
            fl_ctx.set_prop(FLContextKey.RUNNER, client_runner, private=True)

            # # Start the thread for responding the inquire
            # federated_client.stop_listen = False
            # thread = threading.Thread(target=listen_command, args=[federated_client, int(listen_port), client_runner])
            # thread.start()
            # Start the command agent
            command_agent = CommandAgent(federated_client, int(listen_port), client_runner)
            command_agent.start(fl_ctx)

        federated_client.status = ClientStatus.STARTED
        client_runner.run(app_root, args)

    except BaseException as e:
        traceback.print_exc()
        print("FL client execution exception: " + str(e))
    finally:
        # if federated_client:
        #     federated_client.stop_listen = True
        #     thread.join()
        if command_agent:
            command_agent.shutdown()
        if deployer:
            deployer.close()
        # address = ('localhost', 6000)
        # conn_client = Client(address, authkey='client process secret password'.encode())
        # conn_client.send('bye')


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


if __name__ == "__main__":
    """
    This is the program when starting the child process for running the NVIDIA FLARE executor.
    """

    main()

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

import argparse
import logging
import os
import sys
import threading
import time
from multiprocessing.connection import Listener

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, WorkspaceConstants
from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.multi_process_executor_constants import CommunicationMetaData
from nvflare.fuel.hci.server.authz import AuthorizationService
from nvflare.fuel.sec.audit import AuditService
from nvflare.private.defs import EngineConstant
from nvflare.private.fed.app.client.worker_process import check_parent_alive
from nvflare.private.fed.client.admin import FedAdminAgent
from nvflare.private.fed.client.admin_msg_sender import AdminMessageSender
from nvflare.private.fed.client.client_json_config import ClientJsonConfigurator
from nvflare.private.fed.client.client_req_processors import ClientRequestProcessors
from nvflare.private.fed.client.client_run_manager import ClientRunManager
from nvflare.private.fed.client.client_runner import ClientRunner
from nvflare.private.fed.client.fed_client import FederatedClient
from nvflare.private.fed.simulator.simulator_client_engine import SimulatorClientEngine
from nvflare.private.fed.simulator.simulator_const import SimulatorConstants
from nvflare.private.fed.utils.fed_utils import add_logfile_handler
from nvflare.security.security import EmptyAuthorizer


class ClientTaskWorker(FLComponent):
    def create_admin_agent(self, server_args, federated_client: FederatedClient, args, rank=0):
        sender = AdminMessageSender(
            client_name=federated_client.token,
            server_args=server_args,
            secure=False,
        )
        client_engine = SimulatorClientEngine(federated_client, federated_client.token, sender, args, rank)
        admin_agent = FedAdminAgent(
            client_name="admin_agent",
            sender=sender,
            app_ctx=client_engine,
        )
        admin_agent.app_ctx.set_agent(admin_agent)
        federated_client.set_client_engine(client_engine)
        for processor in ClientRequestProcessors.request_processors:
            admin_agent.register_processor(processor)

        client_engine.fire_event(EventType.SYSTEM_START, client_engine.new_context())

        return admin_agent

    def create_client_runner(self, client):
        """Create the ClientRunner for the client to run the ClientApp.

        Args:
            client: the client to run

        """
        app_client_root = client.app_client_root
        args = client.args
        client_config_file_name = os.path.join(app_client_root, args.client_config)
        conf = ClientJsonConfigurator(
            config_file_name=client_config_file_name,
        )
        conf.configure()
        workspace = Workspace(args.workspace, client.client_name, args.config_folder)
        run_manager = ClientRunManager(
            client_name=client.client_name,
            job_id=args.job_id,
            workspace=workspace,
            client=client,
            components=conf.runner_config.components,
            handlers=conf.runner_config.handlers,
            conf=conf,
        )
        client.run_manager = run_manager
        with run_manager.new_context() as fl_ctx:
            fl_ctx.set_prop(FLContextKey.CLIENT_NAME, client.client_name, private=False)
            fl_ctx.set_prop(EngineConstant.FL_TOKEN, client.token, private=False)
            fl_ctx.set_prop(FLContextKey.WORKSPACE_ROOT, args.workspace, private=True)
            fl_ctx.set_prop(FLContextKey.ARGS, args, sticky=True)
            fl_ctx.set_prop(FLContextKey.APP_ROOT, app_client_root, private=True, sticky=True)
            fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True)
            fl_ctx.set_prop(FLContextKey.SECURE_MODE, False, private=True, sticky=True)

            client_runner = ClientRunner(config=conf.runner_config, job_id=args.job_id, engine=run_manager)
            client_runner.init_run(app_client_root, args)
            run_manager.add_handler(client_runner)
            fl_ctx.set_prop(FLContextKey.RUNNER, client_runner, private=True)

            # self.start_command_agent(args, client_runner, federated_client, fl_ctx)

    def do_one_task(self, client):
        stop_run = False
        # Create the ClientRunManager and ClientRunner for the new client to run
        if client.run_manager is None:
            self.create_client_runner(client)
            self.logger.info(f"Initialize ClientRunner for client: {client.client_name}")
        with client.run_manager.new_context() as fl_ctx:
            client_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
            self.fire_event(EventType.SWAP_IN, fl_ctx)

            interval, task_processed = client_runner.run_one_task(fl_ctx)
            self.logger.info(f"Finished one task run for client: {client.client_name}")

            # if any client got the END_RUN event, stop the simulator run.
            if client_runner.end_run_fired or client_runner.asked_to_stop:
                stop_run = True
                self.logger.info("End the Simulator run.")

        return interval, stop_run

    def release_resources(self, client):
        with client.run_manager.new_context() as fl_ctx:
            self.fire_event(EventType.SWAP_OUT, fl_ctx)

            fl_ctx.set_prop(FLContextKey.RUNNER, None, private=True)
        self.logger.info(f"Clean up ClientRunner for : {client.client_name} ")

    def run(self, args, conn):
        admin_agent = None
        try:
            data = conn.recv()
            client = data[SimulatorConstants.CLIENT]
            client_config = data[SimulatorConstants.CLIENT_CONFIG]
            deploy_args = data[SimulatorConstants.DEPLOY_ARGS]

            client.initialize_fobs()

            app_root = os.path.join(args.workspace, SimulatorConstants.JOB_NAME, "app_" + client.client_name)
            app_custom_folder = os.path.join(app_root, "custom")
            sys.path.append(app_custom_folder)

            servers = [{t["name"]: t["service"]} for t in client_config.get("servers")]
            admin_agent = self.create_admin_agent(sorted(servers)[0], client, deploy_args)
            admin_agent.start()
            while True:
                interval, stop_run = self.do_one_task(client)
                conn.send(stop_run)

                continue_run = conn.recv()
                if not continue_run:
                    self.release_resources(client)
                    break
                time.sleep(interval)

        except EOFError:
            self.logger.info("Client run completed.")
        except BaseException as error:
            self.logger.error(f"ClientTaskWorker run error. {error}")
        finally:
            if admin_agent:
                admin_agent.shutdown()


def _create_connection(listen_port):
    address = ("localhost", int(listen_port))
    listener = Listener(address, authkey=CommunicationMetaData.CHILD_PASSWORD.encode())
    conn = listener.accept()
    return conn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-o", type=str, help="WORKSPACE folder", required=True)
    parser.add_argument("--client", type=str, help="Client name", required=True)
    parser.add_argument("--port", type=str, help="Listen port", required=True)
    parser.add_argument("--gpu", "-g", type=str, help="gpu index number")
    parser.add_argument("--parent_pid", type=int, help="parent process pid", required=True)
    args = parser.parse_args()

    log_config_file_path = os.path.join(args.workspace, "startup", "log.config")
    if not os.path.isfile(log_config_file_path):
        log_config_file_path = os.path.join(os.path.dirname(__file__), "resource/log.config")
    logging.config.fileConfig(fname=log_config_file_path, disable_existing_loggers=False)
    log_file = os.path.join(args.workspace, SimulatorConstants.JOB_NAME, "log.txt")
    add_logfile_handler(log_file)

    workspace = os.path.join(args.workspace, SimulatorConstants.JOB_NAME, "app_" + args.client)
    os.chdir(workspace)
    AuthorizationService.initialize(EmptyAuthorizer())
    AuditService.initialize(audit_file_name=WorkspaceConstants.AUDIT_LOG)

    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    conn = _create_connection(args.port)

    # start parent process checking thread
    parent_pid = args.parent_pid
    stop_event = threading.Event()
    thread = threading.Thread(target=check_parent_alive, args=(parent_pid, stop_event))
    thread.start()

    try:
        task_worker = ClientTaskWorker()
        task_worker.run(args, conn)
    finally:
        stop_event.set()
        conn.close()
        AuditService.close()


if __name__ == "__main__":
    """
    This is the main program of simulator worker process when running the NVFlare Simulator..
    """

    main()
    time.sleep(2)
    os._exit(0)

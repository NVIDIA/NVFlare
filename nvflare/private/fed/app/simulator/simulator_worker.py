# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import logging.config
import os
import sys
import threading
import time
from multiprocessing.connection import Listener

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, WorkspaceConstants
from nvflare.fuel.common.multi_process_executor_constants import CommunicationMetaData
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.mpm import MainProcessMonitor as mpm
from nvflare.fuel.hci.server.authz import AuthorizationService
from nvflare.fuel.sec.audit import AuditService
from nvflare.private.fed.app.deployer.base_client_deployer import BaseClientDeployer
from nvflare.private.fed.app.utils import check_parent_alive
from nvflare.private.fed.client.client_engine import ClientEngine
from nvflare.private.fed.client.client_status import ClientStatus
from nvflare.private.fed.client.fed_client import FederatedClient
from nvflare.private.fed.simulator.simulator_app_runner import SimulatorClientAppRunner
from nvflare.private.fed.simulator.simulator_audit import SimulatorAuditor
from nvflare.private.fed.simulator.simulator_const import SimulatorConstants
from nvflare.private.fed.utils.fed_utils import add_logfile_handler, fobs_initialize
from nvflare.security.logging import secure_format_exception
from nvflare.security.security import EmptyAuthorizer

CELL_CONNECT_CHECK_TIMEOUT = 10.0
FETCH_TASK_RUN_RETRY = 3


class ClientTaskWorker(FLComponent):
    def create_client_engine(self, federated_client: FederatedClient, args, rank=0):
        client_engine = ClientEngine(federated_client, federated_client.token, args, rank)
        federated_client.set_client_engine(client_engine)
        federated_client.run_manager = None

        client_engine.fire_event(EventType.SYSTEM_START, client_engine.new_context())

    def create_client_runner(self, client):
        """Create the ClientRunner for the client to run the ClientApp.

        Args:
            client: the client to run

        """
        app_client_root = client.app_client_root
        args = client.args
        args.client_name = client.client_name
        args.token = client.token

        client_app_runner = SimulatorClientAppRunner()
        client_app_runner.client_runner = client_app_runner.create_client_runner(
            app_client_root, args, args.config_folder, client, False
        )
        client_runner = client_app_runner.client_runner
        with client_runner.engine.new_context() as fl_ctx:
            client_app_runner.start_command_agent(args, client, fl_ctx)
            client_app_runner.sync_up_parents_process(client)
            client_runner.engine.cell = client.cell
            client_runner.init_run(app_client_root, args)

    def do_one_task(self, client):
        stop_run = False
        # Create the ClientRunManager and ClientRunner for the new client to run
        try:
            if client.run_manager is None:
                self.create_client_runner(client)
                self.logger.info(f"Initialize ClientRunner for client: {client.client_name}")
            with client.run_manager.new_context() as fl_ctx:
                client_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
                self.fire_event(EventType.SWAP_IN, fl_ctx)

                run_task_tries = 0
                while True:
                    interval, task_processed = client_runner.fetch_and_run_one_task(fl_ctx)
                    if task_processed:
                        self.logger.info(
                            f"Finished one task run for client: {client.client_name} "
                            f"interval: {interval} task_processed: {task_processed}"
                        )

                    # if any client got the END_RUN event, stop the simulator run.
                    if client_runner.end_run_fired or client_runner.asked_to_stop:
                        stop_run = True
                        self.logger.info("End the Simulator run.")
                        break
                    else:
                        if task_processed:
                            break
                        else:
                            run_task_tries += 1
                            if run_task_tries >= FETCH_TASK_RUN_RETRY:
                                break
                            time.sleep(0.5)
        except Exception as e:
            self.logger.error(f"do_one_task execute exception: {secure_format_exception(e)}")
            interval = 1.0
            stop_run = True

        return interval, stop_run

    def release_resources(self, client):
        if client.run_manager:
            with client.run_manager.new_context() as fl_ctx:
                self.fire_event(EventType.SWAP_OUT, fl_ctx)

                fl_ctx.set_prop(FLContextKey.RUNNER, None, private=True)
        self.logger.info(f"Clean up ClientRunner for : {client.client_name} ")

    def run(self, args, conn):
        self.logger.info("ClientTaskWorker started to run")
        admin_agent = None
        client = None
        try:
            data = conn.recv()
            client_config = data[SimulatorConstants.CLIENT_CONFIG]
            deploy_args = data[SimulatorConstants.DEPLOY_ARGS]
            build_ctx = data[SimulatorConstants.BUILD_CTX]

            client = self._create_client(args, build_ctx, deploy_args)

            app_root = os.path.join(args.workspace, SimulatorConstants.JOB_NAME, "app_" + client.client_name)
            app_custom_folder = os.path.join(app_root, "custom")
            sys.path.append(app_custom_folder)

            self.create_client_engine(client, deploy_args)

            while True:
                interval, stop_run = self.do_one_task(client)
                conn.send(stop_run)

                continue_run = conn.recv()
                if not continue_run:
                    self.release_resources(client)
                    break
                time.sleep(interval)

        except Exception as e:
            self.logger.error(f"ClientTaskWorker run error: {secure_format_exception(e)}")
        finally:
            if client:
                client.cell.stop()
            if admin_agent:
                admin_agent.shutdown()

    def _create_client(self, args, build_ctx, deploy_args):
        deployer = BaseClientDeployer()
        deployer.build(build_ctx)
        client = deployer.create_fed_client(deploy_args)

        client.token = args.token
        self._set_client_status(client, deploy_args, args.simulator_root)
        start = time.time()
        self._create_client_cell(client, args.root_url, args.parent_url)
        self.logger.debug(f"Complete _create_client_cell.  Time to create client job cell: {time.time() - start}")
        return client

    def _set_client_status(self, client, deploy_args, simulator_root):
        app_client_root = os.path.join(simulator_root, "app_" + client.client_name)
        client.app_client_root = app_client_root
        client.args = deploy_args
        # self.create_client_runner(client)
        client.simulate_running = False
        client.status = ClientStatus.STARTED

    def _create_client_cell(self, federated_client, root_url, parent_url):
        fqcn = FQCN.join([federated_client.client_name, SimulatorConstants.JOB_NAME])
        credentials = {}
        parent_url = None
        cell = Cell(
            fqcn=fqcn,
            root_url=root_url,
            secure=False,
            credentials=credentials,
            create_internal_listener=False,
            parent_url=parent_url,
        )
        cell.start()
        mpm.add_cleanup_cb(cell.stop)
        federated_client.cell = cell
        federated_client.communicator.cell = cell

        start = time.time()
        while not cell.is_cell_connected(FQCN.ROOT_SERVER):
            time.sleep(0.1)
            if time.time() - start > CELL_CONNECT_CHECK_TIMEOUT:
                raise RuntimeError("Could not connect to the server cell.")


def _create_connection(listen_port):
    address = ("localhost", int(listen_port))
    listener = Listener(address, authkey=CommunicationMetaData.CHILD_PASSWORD.encode())
    conn = listener.accept()
    return conn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-o", type=str, help="WORKSPACE folder", required=True)
    parser.add_argument("--client", type=str, help="Client name", required=True)
    parser.add_argument("--token", type=str, help="Client token", required=True)
    parser.add_argument("--port", type=str, help="Listen port", required=True)
    parser.add_argument("--gpu", "-g", type=str, help="gpu index number")
    parser.add_argument("--parent_pid", type=int, help="parent process pid", required=True)
    parser.add_argument("--simulator_root", "-root", type=str, help="Simulator root folder")
    parser.add_argument("--root_url", "-r", type=str, help="cellnet root_url")
    parser.add_argument("--parent_url", "-p", type=str, help="cellnet parent_url")
    args = parser.parse_args()

    # start parent process checking thread
    parent_pid = args.parent_pid
    stop_event = threading.Event()
    thread = threading.Thread(target=check_parent_alive, args=(parent_pid, stop_event))
    thread.start()

    log_config_file_path = os.path.join(args.workspace, "startup", WorkspaceConstants.LOGGING_CONFIG)
    if not os.path.isfile(log_config_file_path):
        log_config_file_path = os.path.join(os.path.dirname(__file__), WorkspaceConstants.LOGGING_CONFIG)
    logging.config.fileConfig(fname=log_config_file_path, disable_existing_loggers=False)
    workspace = os.path.join(args.workspace, SimulatorConstants.JOB_NAME, "app_" + args.client)
    log_file = os.path.join(workspace, WorkspaceConstants.LOG_FILE_NAME)
    add_logfile_handler(log_file)

    os.chdir(workspace)
    fobs_initialize()
    AuthorizationService.initialize(EmptyAuthorizer())
    # AuditService.initialize(audit_file_name=WorkspaceConstants.AUDIT_LOG)
    AuditService.the_auditor = SimulatorAuditor()

    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    conn = _create_connection(args.port)

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

    # main()
    mpm.run(main_func=main)
    time.sleep(2)
    # os._exit(0)

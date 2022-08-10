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

import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, MachineStatus, WorkspaceConstants
from nvflare.apis.job_def import ALL_SITES, JobMetaKey
from nvflare.apis.workspace import Workspace
from nvflare.fuel.hci.server.authz import AuthorizationService
from nvflare.fuel.hci.zip_utils import convert_legacy_zip, split_path, unzip_all_from_bytes, zip_directory_to_bytes
from nvflare.fuel.sec.audit import AuditService
from nvflare.private.defs import AppFolderConstants, EngineConstant
from nvflare.private.fed.app.deployer.simulator_deployer import SimulatorDeploy
from nvflare.private.fed.client.client_json_config import ClientJsonConfigurator
from nvflare.private.fed.client.client_run_manager import ClientRunManager
from nvflare.private.fed.client.client_runner import ClientRunner
from nvflare.private.fed.client.client_status import ClientStatus
from nvflare.private.fed.server.job_meta_validator import JobMetaValidator
from nvflare.private.fed.simulator.simulator_app_runner import SimulatorServerAppRunner
from nvflare.security.security import EmptyAuthorizer


class SimulatorRunner(FLComponent):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ask_to_stop = False
        self.run_client_index = -1

        self.simulator_root = None
        self.services = None
        self.federated_clients = []
        self.deployer = SimulatorDeploy()

    def setup(self):
        client_names = []
        if self.args.client_list:
            client_names = self.args.client_list.strip().split(",")
        elif self.args.clients:
            for i in range(self.args.clients):
                client_names.append("client" + str(i))

        log_config_file_path = os.path.join(self.args.workspace, "startup", "log.config")
        if not os.path.isfile(log_config_file_path):
            log_config_file_path = os.path.join(os.path.dirname(__file__), "resource/log.config")
        logging.config.fileConfig(fname=log_config_file_path, disable_existing_loggers=False)

        # self.logger = logging.getLogger()
        self.args.log_config = None
        self.args.config_folder = "config"
        self.args.job_id = "simulate_job"
        self.args.client_config = os.path.join(self.args.config_folder, "config_fed_client.json")
        self.args.env = os.path.join("config", AppFolderConstants.CONFIG_ENV)

        os.chdir(self.args.workspace)
        AuthorizationService.initialize(EmptyAuthorizer())
        AuditService.initialize(audit_file_name=WorkspaceConstants.AUDIT_LOG)

        self.simulator_root = os.path.join(self.args.workspace, "simulate_job")
        if os.path.exists(self.simulator_root):
            shutil.rmtree(self.simulator_root)

        try:
            # Validate the simulate job
            job_name = split_path(self.args.job_folder)[1]
            data = zip_directory_to_bytes("", self.args.job_folder)
            data_bytes = convert_legacy_zip(data)
            job_validator = JobMetaValidator()
            valid, error, meta = job_validator.validate(job_name, data_bytes)

            if not valid:
                raise RuntimeError(error)

            if not client_names:
                client_names = self._extract_client_names_from_meta(meta)
            if not client_names:
                self.logger.error("Please provide the client names list to run the simulator")
                sys.exit(1)
            if self.args.threads > len(client_names):
                logging.error("The number of threads to run can not be larger then the number of clients.")
                sys.exit(-1)

            self._validate_client_names(meta, client_names)

            # Deploy the FL server
            self.logger.info("Create the Simulator Server.")
            simulator_server, self.services = self.deployer.create_fl_server(self.args)
            self.services.deploy(self.args, grpc_args=simulator_server)

            # Deploy the FL clients
            self.logger.info("Create the simulate clients.")
            for client_name in client_names:
                self.federated_clients.append(self.deployer.create_fl_client(client_name, self.args))

            self.logger.info("Deploy the Apps.")
            self._deploy_apps(job_name, data_bytes, meta)

            self.logger.info("Set the client status ready.")
            self._set_client_status()
            return True

        except BaseException as error:
            self.logger.error(error)
            return False

    def _extract_client_names_from_meta(self, meta):
        client_names = []
        for _, participants in meta.get(JobMetaKey.DEPLOY_MAP).items():
            for p in participants:
                if p.upper() != ALL_SITES and p != "server":
                    client_names.append(p)
        return client_names

    def _validate_client_names(self, meta, client_names):
        no_app_clients = []
        for name in client_names:
            name_matched = False
            for app_name, participants in meta.get(JobMetaKey.DEPLOY_MAP).items():
                if len(participants) == 1 and participants[0].upper() == ALL_SITES:
                    name_matched = True
                    break
                if name in participants:
                    name_matched = True
                    break
            if not name_matched:
                no_app_clients.append(name)
        if no_app_clients:
            raise RuntimeError(f"The job does not have App to run for clients: {no_app_clients}")

    def _deploy_apps(self, job_name, data_bytes, meta):
        with tempfile.TemporaryDirectory() as temp_dir:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.mkdir(temp_dir)
            unzip_all_from_bytes(data_bytes, temp_dir)
            temp_job_folder = os.path.join(temp_dir, job_name)

            app_server_root = os.path.join(self.simulator_root, "app_server")
            for app_name, participants in meta.get(JobMetaKey.DEPLOY_MAP).items():
                if len(participants) == 1 and participants[0].upper() == ALL_SITES:
                    participants = ["server"]
                    participants.extend([client.client_name for client in self.federated_clients])

                for p in participants:
                    if p == "server":
                        app = os.path.join(temp_job_folder, app_name)
                        shutil.copytree(app, app_server_root)
                    else:
                        app_client_root = os.path.join(self.simulator_root, "app_" + p)
                        app = os.path.join(temp_job_folder, app_name)
                        shutil.copytree(app, app_client_root)

    def run(self):
        try:
            self.logger.info("Deploy and start the Server App.")
            server_thread = threading.Thread(target=self.start_server_app, args=[])
            server_thread.start()

            # wait for the server app is started
            while self.services.engine.engine_info.status != MachineStatus.STARTED:
                time.sleep(1.0)

            self.logger.info("Start the clients run simulation.")
            executor = ThreadPoolExecutor(max_workers=self.args.threads)
            lock = threading.Lock()
            for i in range(self.args.threads):
                executor.submit(lambda p: self.run_client_thread(*p), [self, self.args.threads, lock])

            # wait for the server and client running thread to finish.
            executor.shutdown()
            server_thread.join()
        except BaseException as error:
            self.logger.error(error)
        finally:
            for client in self.federated_clients:
                client.engine.shutdown()
            self.deployer.close()

    def start_server_app(self):
        app_server_root = os.path.join(self.simulator_root, "app_server")
        self.args.server_config = os.path.join("config", AppFolderConstants.CONFIG_FED_SERVER)
        app_custom_folder = os.path.join(app_server_root, "custom")
        sys.path.append(app_custom_folder)

        server_app_runner = SimulatorServerAppRunner()
        snapshot = None
        server_app_runner.start_server_app(
            self.services, self.args, app_server_root, self.args.job_id, snapshot, self.logger
        )

    def _set_client_status(self):
        for client in self.federated_clients:
            app_client_root = os.path.join(self.simulator_root, "app_" + client.client_name)
            client.app_client_root = app_client_root
            client.args = self.args
            # self.create_client_runner(client)
            client.simulate_running = False
            client.status = ClientStatus.STARTED

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

    def get_client_app_name(self, client, meta):
        for app_name, participants in meta.get(JobMetaKey.DEPLOY_MAP).items():
            for p in participants:
                if p == client.client_name:
                    return app_name

        # if client_name not in the deployment map, return the last app
        return app_name

    def run_client_thread(self, simulator_runner, num_of_threads, lock):
        stop_run = False
        interval = 0
        last_run_client_index = -1  # indicates the last run client index

        while not stop_run:
            time.sleep(interval)
            with lock:
                if num_of_threads != len(self.federated_clients) or last_run_client_index == -1:
                    client = self.get_next_run_client()

                    # if the last run_client is not the next one to run again, clear the run_manager and ClientRunner to
                    # release the memory and resources.
                    if self.run_client_index != last_run_client_index and last_run_client_index != -1:
                        self.release_last_run_resources(last_run_client_index)

                    last_run_client_index = self.run_client_index

            client.simulate_running = True
            # Create the ClientRunManager and ClientRunner for the new client to run
            if client.run_manager is None:
                simulator_runner.create_client_runner(client)
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
            client.simulate_running = False

    def release_last_run_resources(self, last_run_client_index):
        last_run_client = self.federated_clients[last_run_client_index]
        with last_run_client.run_manager.new_context() as fl_ctx:
            self.fire_event(EventType.SWAP_OUT, fl_ctx)

            fl_ctx.set_prop(FLContextKey.RUNNER, None, private=True)
            last_run_client.run_manager = None
        self.logger.info(f"Clean up ClientRunner for : {last_run_client.client_name} ")

    def get_next_run_client(self):
        # Find the next client which is not currently running
        while True:
            self.run_client_index = (self.run_client_index + 1) % len(self.federated_clients)
            client = self.federated_clients[self.run_client_index]
            if not client.simulate_running:
                break
        self.logger.info(f"Simulate Run client: {client.client_name}")
        return client

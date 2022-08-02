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

import json
import os
import shutil
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, MachineStatus
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.workspace import Workspace
from nvflare.private.defs import AppFolderConstants
from nvflare.private.defs import EngineConstant
from nvflare.private.fed.client.client_json_config import ClientJsonConfigurator
from nvflare.private.fed.client.client_run_manager import ClientRunManager
from nvflare.private.fed.client.client_runner import ClientRunner
from nvflare.private.fed.client.client_status import ClientStatus
from nvflare.private.fed.simulator.simulator_client_app_runner import SimulatorServerAppRunner


class SimulatorRunner(FLComponent):

    def __init__(self):
        super().__init__()
        self.ask_to_stop = False
        self.run_client_index = -1

    def run(self, simulator_root, args, logger, services, federated_clients):
        # TODO: Add a job validator to check the job validation.

        meta_file = os.path.join(args.job_folder, "meta.json")
        with open(meta_file, "rb") as f:
            meta_data = f.read()
        meta = json.loads(meta_data)

        self.logger.info("Deploy and start the Server App.")
        server_thread = threading.Thread(target=self.start_server_app,
                                         args=[simulator_root, args, logger, services, meta])
        server_thread.start()

        # wait for the server app is started
        while services.engine.engine_info.status != MachineStatus.STARTED:
            time.sleep(1.0)

        self.logger.info("Deploy Client Apps.")
        self.deploy_client_apps(simulator_root, args, federated_clients, meta)

        self.logger.info("Start the clients run simulation.")
        executor = ThreadPoolExecutor(max_workers=args.threads)
        lock = threading.Lock()
        for i in range(args.threads):
            executor.submit(lambda p: self.run_client_thread(*p),
                            [self, args.threads, federated_clients, lock, self.logger])

        # wait for the server and client running thread to finish.
        executor.shutdown()
        server_thread.join()

    def start_server_app(self, simulator_root, args, logger, services, meta):
        app_server_root = os.path.join(simulator_root, "app_server")
        for app_name, participants in meta.get(JobMetaKey.DEPLOY_MAP).items():
            for p in participants:
                if p == "server":
                    app = os.path.join(args.job_folder, app_name)
                    shutil.copytree(app, app_server_root)

        args.server_config = os.path.join("config", AppFolderConstants.CONFIG_FED_SERVER)
        app_custom_folder = os.path.join(app_server_root, "custom")
        sys.path.append(app_custom_folder)

        server_app_runner = SimulatorServerAppRunner()
        snapshot = None
        server_app_runner.start_server_app(services, args, app_server_root, args.job_id, snapshot, logger)

    def deploy_client_apps(self, simulator_root, args, federated_clients, meta):
        for client in federated_clients:
            app_client_root = os.path.join(simulator_root, "app_" + client.client_name)

            app_name = self.get_client_app_name(client, meta)
            app = os.path.join(args.job_folder, app_name)
            shutil.copytree(app, app_client_root)

            client.app_client_root = app_client_root
            client.args = args
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

    def run_client_thread(self, simulator_runner, num_of_threads, federated_clients, lock, logger):
        stop_run = False
        interval = 0
        last_run_client_index = -1  # indicates the last run client index

        while not stop_run:
            time.sleep(interval)
            with lock:
                if num_of_threads != len(federated_clients) or last_run_client_index == -1:
                    client = self.get_next_run_client(federated_clients, logger)

                    # if the last run_client is not the next one to run again, clear the run_manager and ClientRunner to
                    # release the memory and resources.
                    if self.run_client_index != last_run_client_index and last_run_client_index != -1:
                        self.release_last_run_resources(federated_clients, last_run_client_index, logger)

                    last_run_client_index = self.run_client_index

            client.simulate_running = True
            # Create the ClientRunManager and ClientRunner for the new client to run
            if client.run_manager is None:
                simulator_runner.create_client_runner(client)
                logger.info(f"Initialize ClientRunner for client: {client.client_name}")

            with client.run_manager.new_context() as fl_ctx:
                client_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
                client_runner.fire_event(EventType.SWAP_IN, fl_ctx)

                interval, task_processed = client_runner.run_one_task(fl_ctx)
                logger.info(f"Finished one task run for client: {client.client_name}")

                # if any client got the END_RUN event, stop the simulator run.
                if client_runner.end_run_fired or client_runner.asked_to_stop:
                    stop_run = True
                    logger.info("End the Simulator run.")
            client.simulate_running = False

    def release_last_run_resources(self, federated_clients, last_run_client_index, logger):
        last_run_client = federated_clients[last_run_client_index]
        with last_run_client.run_manager.new_context() as fl_ctx:
            client_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
            client_runner.fire_event(EventType.SWAP_OUT, fl_ctx)

            fl_ctx.set_prop(FLContextKey.RUNNER, None, private=True)
            last_run_client.run_manager = None
        logger.info(f"Clean up ClientRunner for : {last_run_client.client_name} ")

    def get_next_run_client(self, federated_clients, logger):
        # Find the next client which is not currently running
        while True:
            self.run_client_index = (self.run_client_index + 1) % len(federated_clients)
            client = federated_clients[self.run_client_index]
            if not client.simulate_running:
                break
        logger.info(f"Simulate Run client: {client.client_name}")
        return client

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
from nvflare.apis.fl_constant import FLContextKey
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

    def run(self, simulator_root, args, logger, services, federated_clients):
        meta_file = os.path.join(args.job_folder, "meta.json")
        with open(meta_file, "rb") as f:
            meta_data = f.read()
        meta = json.loads(meta_data)

        threading.Thread(target=self.start_server, args=[simulator_root, args, logger, services, meta]).start()

        time.sleep(5.0)

        self.create_client_run_managers(simulator_root, args, federated_clients, meta)

        executor = ThreadPoolExecutor(max_workers=args.threads)
        run_client_index = 0
        lock = threading.Lock()
        for i in range(args.threads):
            executor.submit(lambda p: start_client(*p), [self, federated_clients, run_client_index, lock])
            # threading.Thread(target=self.start_client, args=[simulator_root, args, federated_clients[0], meta]).start()

    def start_server(self, simulator_root, args, logger, services, meta):
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

    def create_client_run_managers(self, simulator_root, args, federated_clients, meta):
        for client in federated_clients:
            app_client_root = os.path.join(simulator_root, "app_" + client.client_name)

            app_name = self.get_client_app_name(client, meta)
            app = os.path.join(args.job_folder, app_name)
            shutil.copytree(app, app_client_root)

            client.app_client_root = app_client_root
            client.args = args
            # self.create_client_runner(client)
            client.status = ClientStatus.STARTED

    def create_client_runner(self, client):
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

    # def start_client(self, simulator_root, args, federated_client, meta):
    #     for app_name, participants in meta.get(JobMetaKey.DEPLOY_MAP).items():
    #         for p in participants:
    #             if p != "server":
    #                 app_client_root = os.path.join(simulator_root, "app_" + p)
    #                 app = os.path.join(args.job_folder, app_name)
    #                 shutil.copytree(app, app_client_root)
    #
    #                 args.client_name = p
    #                 args.token = federated_client.token
    #                 client_app_runner = SimulatorClientAppRunner()
    #                 client_app_runner.start_run(app_client_root, args, args.config_folder, federated_client, False)


def start_client(simulator_runner, federated_clients, run_client_index, lock):
    stop_run = False
    interval = 0
    last_run_client_index = -1
    while not stop_run:
        time.sleep(interval)
        with lock:
            client = federated_clients[run_client_index]

            if run_client_index != last_run_client_index and last_run_client_index != -1:
                last_run_client = federated_clients[last_run_client_index]
                with last_run_client.run_manager.new_context() as fl_ctx:
                    fl_ctx.set_prop(FLContextKey.RUNNER, None, private=True)
                    last_run_client.run_manager = None
                    client_runner.fire_event(EventType.END_RUN, fl_ctx)

            last_run_client_index = run_client_index
            run_client_index = (run_client_index + 1) % len(federated_clients)

        if client.run_manager is None:
            simulator_runner.create_client_runner(client)
        with client.run_manager.new_context() as fl_ctx:
            client_runner = fl_ctx.get_prop(FLContextKey.RUNNER)

            task_processed = False
            while not task_processed:
                interval, task_processed = client_runner.run_one_round(fl_ctx)

                # if any client got the END_RUN event, stop the simulator run.
                if client_runner.end_run_fired or client_runner.asked_to_stop:
                    stop_run = True
                    break

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
import logging
import os

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.private.defs import EngineConstant
from nvflare.private.fed.app.fl_conf import create_privacy_manager
from nvflare.private.fed.client.client_json_config import ClientJsonConfigurator
from nvflare.private.fed.client.client_run_manager import ClientRunManager
from nvflare.private.fed.client.client_runner import ClientRunner
from nvflare.private.fed.client.client_status import ClientStatus
from nvflare.private.fed.client.command_agent import CommandAgent
from nvflare.private.fed.runner import Runner
from nvflare.private.fed.utils.fed_utils import authorize_build_component
from nvflare.private.privacy_manager import PrivacyService


class ClientAppRunner(Runner):

    logger = logging.getLogger("ClientAppRunner")

    def __init__(self, time_out=60.0) -> None:
        super().__init__()
        self.command_agent = None
        self.timeout = time_out
        self.client_runner = None

    def start_run(self, app_root, args, config_folder, federated_client, secure_train, sp, event_handlers):
        self.client_runner = self.create_client_runner(
            app_root, args, config_folder, federated_client, secure_train, event_handlers
        )

        federated_client.set_client_runner(self.client_runner)
        federated_client.set_primary_sp(sp)

        with self.client_runner.engine.new_context() as fl_ctx:
            self.start_command_agent(args, federated_client, fl_ctx)

        self.sync_up_parents_process(federated_client)

        federated_client.start_overseer_agent()
        federated_client.status = ClientStatus.STARTED
        self.client_runner.run(app_root, args)
        federated_client.stop_cell()

    @staticmethod
    def _set_fl_context(fl_ctx: FLContext, app_root, args, workspace, secure_train):
        fl_ctx.set_prop(FLContextKey.CLIENT_NAME, args.client_name, private=False)
        fl_ctx.set_prop(EngineConstant.FL_TOKEN, args.token, private=False)
        fl_ctx.set_prop(FLContextKey.WORKSPACE_ROOT, args.workspace, private=True)
        fl_ctx.set_prop(FLContextKey.ARGS, args, sticky=True)
        fl_ctx.set_prop(FLContextKey.APP_ROOT, app_root, private=True, sticky=True)
        fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True)
        fl_ctx.set_prop(FLContextKey.SECURE_MODE, secure_train, private=True, sticky=True)
        fl_ctx.set_prop(FLContextKey.CURRENT_RUN, args.job_id, private=False, sticky=True)
        fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, args.job_id, private=False, sticky=True)

    def create_client_runner(self, app_root, args, config_folder, federated_client, secure_train, event_handlers=None):
        workspace = Workspace(args.workspace, args.client_name, config_folder)
        fl_ctx = FLContext()
        self._set_fl_context(fl_ctx, app_root, args, workspace, secure_train)
        client_config_file_name = os.path.join(app_root, args.client_config)
        conf = ClientJsonConfigurator(config_file_name=client_config_file_name, args=args, kv_list=args.set)
        if event_handlers:
            conf.set_component_build_authorizer(authorize_build_component, fl_ctx=fl_ctx, event_handlers=event_handlers)
        conf.configure()

        runner_config = conf.runner_config

        # configure privacy control!
        privacy_manager = create_privacy_manager(workspace, names_only=False)
        if privacy_manager.is_policy_defined():
            if privacy_manager.components:
                for cid, comp in privacy_manager.components.items():
                    runner_config.add_component(cid, comp)

        # initialize Privacy Service
        PrivacyService.initialize(privacy_manager)

        run_manager = self.create_run_manager(args, conf, federated_client, workspace)
        federated_client.run_manager = run_manager
        with run_manager.new_context() as fl_ctx:
            self._set_fl_context(fl_ctx, app_root, args, workspace, secure_train)
            client_runner = ClientRunner(config=conf.runner_config, job_id=args.job_id, engine=run_manager)
            run_manager.add_handler(client_runner)
            fl_ctx.set_prop(FLContextKey.RUNNER, client_runner, private=True)

            # self.start_command_agent(args, client_runner, federated_client, fl_ctx)
        return client_runner

    def create_run_manager(self, args, conf, federated_client, workspace):
        return ClientRunManager(
            client_name=args.client_name,
            job_id=args.job_id,
            workspace=workspace,
            client=federated_client,
            components=conf.runner_config.components,
            handlers=conf.runner_config.handlers,
            conf=conf,
        )

    def start_command_agent(self, args, federated_client, fl_ctx):
        # Start the command agent
        self.command_agent = CommandAgent(federated_client)
        self.command_agent.start(fl_ctx)

    def sync_up_parents_process(self, federated_client):
        run_manager = federated_client.run_manager
        with run_manager.new_context() as fl_ctx:
            run_manager.get_all_clients_from_server(fl_ctx)

    def close(self):
        if self.command_agent:
            self.command_agent.shutdown()

    def stop(self):
        if self.client_runner:
            self.client_runner.asked_to_stop = True

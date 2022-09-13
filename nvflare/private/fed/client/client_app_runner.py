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

import os
import sys

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.workspace import Workspace
from nvflare.private.defs import EngineConstant
from nvflare.private.fed.app.fl_conf import create_privacy_manager
from nvflare.private.fed.client.client_json_config import ClientJsonConfigurator
from nvflare.private.fed.client.client_run_manager import ClientRunManager
from nvflare.private.fed.client.client_runner import ClientRunner
from nvflare.private.fed.client.client_status import ClientStatus
from nvflare.private.fed.client.command_agent import CommandAgent
from nvflare.private.privacy_manager import PrivacyService


class ClientAppRunner:
    def __init__(self) -> None:
        self.command_agent = None

    def start_run(self, app_root, args, config_folder, federated_client, secure_train):
        client_runner = self.create_client_runner(app_root, args, config_folder, federated_client, secure_train)
        federated_client.status = ClientStatus.STARTED
        client_runner.run(app_root, args)

    def create_client_runner(self, app_root, args, config_folder, federated_client, secure_train):
        client_config_file_name = os.path.join(app_root, args.client_config)
        conf = ClientJsonConfigurator(
            config_file_name=client_config_file_name,
        )
        conf.configure()
        workspace = Workspace(args.workspace, args.client_name, config_folder)
        app_custom_folder = workspace.get_client_custom_dir()
        if os.path.isdir(app_custom_folder):
            sys.path.append(app_custom_folder)

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
            fl_ctx.set_prop(FLContextKey.CLIENT_NAME, args.client_name, private=False)
            fl_ctx.set_prop(EngineConstant.FL_TOKEN, args.token, private=False)
            fl_ctx.set_prop(FLContextKey.WORKSPACE_ROOT, args.workspace, private=True)
            fl_ctx.set_prop(FLContextKey.ARGS, args, sticky=True)
            fl_ctx.set_prop(FLContextKey.APP_ROOT, app_root, private=True, sticky=True)
            fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True)
            fl_ctx.set_prop(FLContextKey.SECURE_MODE, secure_train, private=True, sticky=True)

            client_runner = ClientRunner(config=conf.runner_config, job_id=args.job_id, engine=run_manager)
            run_manager.add_handler(client_runner)
            fl_ctx.set_prop(FLContextKey.RUNNER, client_runner, private=True)

            self.start_command_agent(args, client_runner, federated_client, fl_ctx)
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

    def start_command_agent(self, args, client_runner, federated_client, fl_ctx):
        # Start the command agent
        self.command_agent = CommandAgent(federated_client, int(args.listen_port), client_runner)
        self.command_agent.start(fl_ctx)

    def close(self):
        if self.command_agent:
            self.command_agent.shutdown()

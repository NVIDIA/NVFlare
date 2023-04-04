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

from nvflare.apis.fl_constant import FLContextKey
from nvflare.private.fed.client.client_app_runner import ClientAppRunner
from nvflare.private.fed.client.client_run_manager import ClientRunManager
from nvflare.private.fed.server.server_app_runner import ServerAppRunner


class SimulatorClientRunManager(ClientRunManager):
    def create_job_processing_context_properties(self, workspace, job_id):
        return {}


class SimulatorClientAppRunner(ClientAppRunner):
    def create_run_manager(self, args, conf, federated_client, workspace):
        run_manager = SimulatorClientRunManager(
            client_name=args.client_name,
            job_id=args.job_id,
            workspace=workspace,
            client=federated_client,
            components=conf.runner_config.components,
            handlers=conf.runner_config.handlers,
            conf=conf,
        )
        with run_manager.new_context() as fl_ctx:
            fl_ctx.set_prop(FLContextKey.SIMULATE_MODE, True, private=True, sticky=True)
        return run_manager


class SimulatorServerAppRunner(ServerAppRunner):
    def __init__(self, server) -> None:
        super().__init__(server)

    def sync_up_parents_process(self, args):
        pass

    def update_job_run_status(self):
        pass

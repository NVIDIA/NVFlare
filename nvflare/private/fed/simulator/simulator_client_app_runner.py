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

from nvflare.private.fed.client.client_app_runner import ClientAppRunner
from nvflare.private.fed.server.server_app_runner import ServerAppRunner
from nvflare.private.fed.client.client_runner import ClientRunner


class SimulatorClientAppRunner(ClientAppRunner):

    def start_command_agent(self, args, client_runner, federated_client, fl_ctx):
        pass

    def start_run(self, app_root, args, config_folder, federated_client, secure_train):
        with run_manager.new_context() as fl_ctx:

            client_runner = ClientRunner(config=conf.runner_config, job_id=args.job_id, engine=run_manager)
            run_manager.add_handler(client_runner)
            fl_ctx.set_prop(FLContextKey.RUNNER, client_runner, private=True)

            self.start_command_agent(args, client_runner, federated_client, fl_ctx)
        federated_client.status = ClientStatus.STARTED
        client_runner.run(app_root, args)


class SimulatorServerAppRunner(ServerAppRunner):

    def sync_up_parents_process(self, args, server):
        pass

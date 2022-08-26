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

from typing import List, Optional

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, RunProcessKey, ServerCommandKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.private.fed.server.server_state import HotState

from ..server.fed_server import FederatedServer
from ..server.server_engine import ServerEngine


class SimulatorServerEngine(ServerEngine):
    def persist_components(self, fl_ctx: FLContext, completed: bool):
        pass

    def sync_clients_from_main_process(self):
        pass

    def parent_aux_send(self, targets: [], topic: str, request: Shareable, timeout: float, fl_ctx: FLContext) -> dict:
        replies = self.aux_send(targets=targets, topic=topic, request=request, timeout=timeout, fl_ctx=fl_ctx)

        return replies


class SimulatorServer(FederatedServer):
    def __init__(
        self,
        project_name=None,
        min_num_clients=2,
        max_num_clients=10,
        cmd_modules=None,
        heart_beat_timeout=600,
        handlers: Optional[List[FLComponent]] = None,
        args=None,
        secure_train=False,
        enable_byoc=False,
        snapshot_persistor=None,
        overseer_agent=None,
    ):
        super().__init__(
            project_name,
            min_num_clients,
            max_num_clients,
            cmd_modules,
            heart_beat_timeout,
            handlers,
            args,
            secure_train,
            enable_byoc,
            snapshot_persistor,
            overseer_agent,
        )

        self.engine.run_processes["simulate_job"] = {
            RunProcessKey.LISTEN_PORT: None,
            RunProcessKey.CONNECTION: None,
            RunProcessKey.CHILD_PROCESS: None,
            RunProcessKey.JOB_ID: "simulate_job",
            # RunProcessKey.PARTICIPANTS: job_clients,
        }

        self.server_state = HotState()

    def _process_task_request(self, client, fl_ctx, shared_fl_ctx: FLContext):
        fl_ctx.set_peer_context(shared_fl_ctx)
        server_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        taskname, task_id, shareable = server_runner.process_task_request(client, fl_ctx)

        return shareable, task_id, taskname

    def _submit_update(self, data, shared_fl_context):
        with self.engine.new_context() as fl_ctx:
            shareable = data.get(ReservedKey.SHAREABLE)
            shared_fl_ctx = data.get(ReservedKey.SHARED_FL_CONTEXT)

            client = shareable.get_header(ServerCommandKey.FL_CLIENT)
            fl_ctx.set_peer_context(shared_fl_ctx)
            contribution_task_name = shareable.get_header(ServerCommandKey.TASK_NAME)
            task_id = shareable.get_cookie(FLContextKey.TASK_ID)
            server_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
            server_runner.process_submission(client, contribution_task_name, task_id, shareable, fl_ctx)

    def remove_dead_clients(self):
        pass

    def _aux_communicate(self, fl_ctx, shareable, shared_fl_context, topic):
        try:
            with self.engine.lock:
                reply = self.engine.dispatch(topic=topic, request=shareable, fl_ctx=fl_ctx)
        except BaseException:
            self.logger.info("Could not connect to server runner process - asked client to end the run")
            reply = make_reply(ReturnCode.COMMUNICATION_ERROR)

        return reply

    def _create_server_engine(self, args, snapshot_persistor):
        return SimulatorServerEngine(
            server=self, args=args, client_manager=self.client_manager, snapshot_persistor=snapshot_persistor
        )

    def deploy(self, args, grpc_args=None, secure_train=False):
        super(FederatedServer, self).deploy(args, grpc_args, secure_train)

    def stop_training(self):
        self.engine.run_processes.clear()
        super().stop_training()

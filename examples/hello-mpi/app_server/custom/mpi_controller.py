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

import traceback

from nvflare.apis.client import Client
from nvflare.apis.collective_comm_constants import CollectiveCommEvent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller, Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal


class MPIController(Controller):
    def __init__(
        self,
        train_timeout: int = 300,
    ):
        """MPI controller.

        Args:
            train_timeout (int, optional): Time to wait for clients to do local training.

        Raises:
            TypeError: when any of input arguments does not have correct type
            ValueError: when any of input arguments is out of range
        """
        super().__init__()

        if not isinstance(train_timeout, int):
            raise TypeError("train_timeout must be int but got {}".format(type(train_timeout)))

        self._train_timeout = train_timeout

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == CollectiveCommEvent.FAILED:
            self.system_panic(reason="Collective communication failed", fl_ctx=fl_ctx)

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.log_info(fl_ctx, "Initializing MPIController workflow.")

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        try:
            self.log_info(fl_ctx, "Begin training phase.")

            # Assumption: all clients are used
            clients = self._engine.get_clients()

            train_task = Task(
                name="mpi_train",
                data=Shareable(),
                timeout=self._train_timeout,
            )

            self.broadcast_and_wait(
                task=train_task,
                targets=clients,
                min_responses=len(clients),
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )

            self.log_info(fl_ctx, "Finish training phase.")

        except BaseException as e:
            err = traceback.format_exc()
            error_msg = f"Exception in control_flow: {e}: {err}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(str(e), fl_ctx)

    def stop_controller(self, fl_ctx: FLContext) -> None:
        self.cancel_all_tasks()

    def process_result_of_unknown_task(
        self, client: Client, task_name, client_task_id, result: Shareable, fl_ctx: FLContext
    ) -> None:
        self.log_warning(fl_ctx, f"Unknown task: {task_name} from client {client.name}.")

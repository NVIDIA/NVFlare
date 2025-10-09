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

import time
from threading import Lock

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.streaming import StreamableEngine
from nvflare.client.config import ExchangeFormat

from .receiver import TensorReceiver
from .sender import TensorSender
from .store import TensorStore
from .types import TensorEventTypes
from .utils import clean_task_data


class TensorServerStreamer(FLComponent):
    """Handles sending task data tensors to clients and receiving task results from clients.

    It uses a StreamableEngine, TensorReceiver, and TensorSender to manage tensor streaming on the server side.
    Attributes:
        format (str): The format of the tensors to send/receive. Default is "pytorch".
        entry_timeout (float): Timeout for tensor entry transfer operations. Default is 30.0 seconds.
        engine (StreamableEngine): The StreamableEngine used for tensor streaming.
        sender (TensorSender): The TensorSender used to send tensors to clients.
        receiver (TensorReceiver): The TensorReceiver used to receive tensors from clients.
        start_sending_time (float): The timestamp when sending to clients started.
        num_task_data_sent (int): The number of task data sent to clients.
        num_task_skipped (int): The number of task data skipped (not sent) to clients.
        data_cleaned (bool): Flag indicating whether the task data has been cleaned from the FLContext.
    Methods:
        initialize(fl_ctx): Initializes the TensorServerStreamer component.
        handle_event(event_type, fl_ctx): Handles events for the TensorSender component.
        send_tensors_to_client(fl_ctx): Sends tensors to the client after task data filtering.
        try_to_clean_task_data(fl_ctx): Cleans the task data in the FLContext if all clients have received the tensors.
    """

    def __init__(
        self,
        format: ExchangeFormat = ExchangeFormat.PYTORCH,
        tasks: list[str] = None,
        entry_timeout: float = 30.0,
        wait_all_clients_timeout: float = 300.0,
    ):
        """Initialize the TensorServerStreamer component.

        Args:
            format (ExchangeFormat): The format of the tensors to send/receive. Default is ExchangeFormat.TORCH.
            tasks (list[str]): The list of tasks to send tensors for. Default is None, which means the "train" task.
            entry_timeout (float): Timeout for tensor entry transfer operations. Default is 10.0 seconds.
            wait_all_clients_timeout (float): Timeout for sending tensors to all clients. Default is 120.0 seconds.
        """
        super().__init__()
        self.format = format
        self.tasks = tasks if tasks is not None else ["train"]
        self.entry_timeout = entry_timeout
        self.wait_all_clients_timeout = wait_all_clients_timeout
        self.engine: StreamableEngine = None
        self.sender: TensorSender = None
        self.receiver: TensorReceiver = None
        self.start_sending_time: float = None
        self.task_to_store: dict[str, TensorStore] = {}
        self.data_cleaned = False
        self.num_task_data_stored = 0
        self.num_task_skipped = 0
        self.lock = Lock()

    def initialize(self, fl_ctx: FLContext):
        """Initialize the TensorServerSender component.
        Args:
            fl_ctx (FLContext): The FLContext for the current operation.
        """
        engine: StreamableEngine = fl_ctx.get_engine()
        if not engine:
            self.system_panic(f"Engine not found. {self.__class__.__name__} exiting.", fl_ctx)
            return

        if not isinstance(engine, StreamableEngine):
            self.system_panic(
                f"Engine is not a StreamableEngine. {self.__class__.__name__} exiting.",
                fl_ctx,
            )
            return

        self.engine = engine
        try:
            self.receiver = TensorReceiver(engine, FLContextKey.TASK_RESULT, self.format)
        except Exception as e:
            self.system_panic(str(e), fl_ctx)
            return

        self.sender = TensorSender(self.engine, FLContextKey.TASK_DATA, self.format)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Handle events for the TensorSender component.

        Args:
            event_type (str): The type of event to handle.
            fl_ctx (FLContext): The FLContext for the current operation.
        """
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.BEFORE_TASK_DATA_FILTER:
            self.reset_counters(fl_ctx)
        elif event_type == EventType.AFTER_TASK_DATA_FILTER:
            self.store_tensors_for_sending(fl_ctx)
            self.wait_all_tasks_to_store_tensors(fl_ctx)
            self.try_to_clean_task_data(fl_ctx)
        elif event_type == TensorEventTypes.SEND_TENSORS_FOR_TASK_DATA:
            recv_event_data: Shareable = fl_ctx.get_prop(FLContextKey.EVENT_DATA)
            peer_name = recv_event_data.get_header("_origin")
            task_id = recv_event_data["task_id"]
            self.log_debug(
                fl_ctx,
                f"Received SEND_TENSORS_FOR_TASK_DATA event, "
                f"starting sending tensors to client '{peer_name}' for task_id '{task_id}'.",
            )
            self.send_tensors_to_client(fl_ctx, task_id, [peer_name])
        elif event_type == EventType.BEFORE_TASK_RESULT_FILTER:
            peer_name = fl_ctx.get_peer_context().get_identity_name()
            while not self.receiver.has_tensors(peer_name):
                self.log_debug(fl_ctx, f"Waiting to receive tensors from peer {peer_name}.")
                time.sleep(0.1)
                if fl_ctx.get_run_abort_signal():
                    return

            self.receiver.set_ctx_with_tensors(fl_ctx)

    def store_tensors_for_sending(self, fl_ctx: FLContext):
        """Store tensors in the TensorStore for sending.

        Args:
            fl_ctx (FLContext): The FLContext for the current operation.
        """
        task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
        task_name = fl_ctx.get_prop(FLContextKey.TASK_NAME)
        store = TensorStore(task_id, task_name, FLContextKey.TASK_DATA)
        store.parse(fl_ctx)
        self.task_to_store[task_id] = store
        with self.lock:
            self.num_task_data_stored += 1

    def reset_counters(self, fl_ctx: FLContext):
        """Reset the counters for the number of task data sent and skipped."""
        with self.lock:
            if self.data_cleaned:
                self.log_debug(fl_ctx, "Resetting counters for task data sent and skipped.")
                self.num_task_data_stored = 0
                self.num_task_skipped = 0
                self.start_sending_time = None
                self.data_cleaned = False

    def send_tensors_to_client(self, fl_ctx: FLContext, task_id: str, targets: list[str]):
        """Send tensors to the client after task data filtering.

        Args:
            fl_ctx (FLContext): The FLContext for the current operation.
        """
        with self.lock:
            if not self.start_sending_time:
                self.start_sending_time = time.time()

        store = self.task_to_store.pop(task_id)
        if not store:
            self.log_warning(fl_ctx, f"No tensors found for task_id: {task_id}. Skipping sending tensors.")
            with self.lock:
                self.num_task_skipped += 1
            return

        try:
            self.sender.send(fl_ctx, store, targets, self.entry_timeout)
        except ValueError as e:
            self.system_panic(f"Failed to send tensors: {e}", fl_ctx)
            raise e

    def wait_all_tasks_to_store_tensors(self, fl_ctx: FLContext):
        """Wait until all client tasks have stored their tensors."""
        if not self.start_sending_time:
            return

        num_clients = len(self.engine.get_clients())

        while True:
            time.sleep(0.1)

            num_processed = self.num_task_data_stored + self.num_task_skipped
            if num_processed >= num_clients:
                return

            if time.time() - self.start_sending_time > self.wait_all_clients_timeout:
                self.system_panic(
                    f"Timeout waiting for all clients to receive tensors. Sent to {self.num_task_data_sent} out of {num_clients},"
                    f" skipped {self.num_task_skipped}.",
                    fl_ctx,
                )
                return

    def try_to_clean_task_data(self, fl_ctx: FLContext):
        """Clean the task data in the FLContext.

        Args:
            fl_ctx (FLContext): The FLContext to clean the task data from.
        """
        num_clients = len(self.engine.get_clients())
        # only clean if we successfully already stored tensors for all clients
        with self.lock:
            if self.num_task_data_stored >= num_clients and not self.data_cleaned:
                clean_task_data(fl_ctx)
                self.data_cleaned = True

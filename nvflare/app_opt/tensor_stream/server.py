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
from nvflare.apis.streaming import StreamableEngine
from nvflare.client.config import ExchangeFormat

from .receiver import TensorReceiver
from .sender import TensorSender
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
        self.num_task_data_sent = 0
        self.num_task_skipped = 0
        self.data_cleaned = False
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

        self.sender = TensorSender(engine, FLContextKey.TASK_DATA, self.format, self.tasks)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Handle events for the TensorSender component.

        Args:
            event_type (str): The type of event to handle.
            fl_ctx (FLContext): The FLContext for the current operation.
        """
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.BEFORE_TASK_DATA_FILTER:
            self.reset_counters()
        elif event_type == EventType.AFTER_TASK_DATA_FILTER:
            num_clients = len(self.engine.get_clients())
            self.send_tensors_to_client(fl_ctx)
            self.wait_clients_to_complete(num_clients, fl_ctx)
            self.try_to_clean_task_data(num_clients, fl_ctx)
        elif event_type == EventType.BEFORE_TASK_RESULT_FILTER:
            self.receiver.set_ctx_with_tensors(fl_ctx)

    def reset_counters(self):
        """Reset the counters for the number of task data sent and skipped."""
        with self.lock:
            if self.data_cleaned:
                self.num_task_data_sent = 0
                self.num_task_skipped = 0
                self.start_sending_time = None
                self.data_cleaned = False

    def send_tensors_to_client(self, fl_ctx: FLContext):
        """Send tensors to the client after task data filtering.

        Args:
            fl_ctx (FLContext): The FLContext for the current operation.
        """
        with self.lock:
            if not self.start_sending_time:
                self.start_sending_time = time.time()

        try:
            success = self.sender.send(fl_ctx, self.entry_timeout)
        except ValueError as e:
            self.system_panic(f"Failed to send tensors: {e}", fl_ctx)
            return

        with self.lock:
            if success:
                self.num_task_data_sent += 1
            else:
                self.num_task_skipped += 1

    def wait_clients_to_complete(self, num_clients: int, fl_ctx: FLContext):
        """Wait until all clients have received the tensors or timeout occurs."""
        if not self.start_sending_time:
            return

        while True:
            time.sleep(0.1)

            num_processed = self.num_task_data_sent + self.num_task_skipped
            if num_processed >= num_clients:
                return

            if time.time() - self.start_sending_time > self.wait_all_clients_timeout:
                self.system_panic(
                    f"Timeout waiting for all clients to receive tensors. Sent to {self.num_task_data_sent} out of {num_clients},"
                    f" skipped {self.num_task_skipped}.",
                    fl_ctx,
                )
                return

    def try_to_clean_task_data(self, num_clients: int, fl_ctx: FLContext):
        """Clean the task data in the FLContext.

        Args:
            fl_ctx (FLContext): The FLContext to clean the task data from.
        """
        # only clean if we successfully sent to all clients
        with self.lock:
            if self.num_task_data_sent >= num_clients and not self.data_cleaned:
                self.log_info(
                    fl_ctx,
                    f"Tensors were sent to all clients, removing them from task data. "
                    f"Sent {self.num_task_data_sent} out of {num_clients}",
                )
                clean_task_data(fl_ctx)
                self.data_cleaned = True

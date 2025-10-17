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
from collections import defaultdict
from threading import Lock

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.streaming import StreamableEngine
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
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
        start_sending_time (dict[int, float]): The timestamp when sending to clients started for the current round.
        seen_tasks (dict[int, set[str]]): The set of task IDs seen in the current round.
        num_task_data_sent (dict[int, int]): The number of task data sent to clients successfully for the current round.
        num_task_skipped (dict[int, int]): The number of task data skipped (not sent) to clients for the current round.
        data_cleaned (dict[int, bool]): Flag indicating whether the task data has been cleaned from the FLContext for the current round.
        lock (Lock): A lock to protect shared data structures.
    Methods:
        initialize(fl_ctx): Initializes the TensorServerStreamer component.
        handle_event(event_type, fl_ctx): Handles events for the TensorSender component.
        send_tensors_to_client(fl_ctx): Sends tensors to the client after task data filtering.
        clean_counters(current_round): Cleans the counters for the current round.
        wait_clients_to_complete(num_clients, fl_ctx): Waits until all clients have received the tensors or timeout occurs.
        try_to_clean_task_data(fl_ctx): Cleans the task data in the FLContext if all clients have received the tensors.
    """

    def __init__(
        self,
        format: ExchangeFormat = ExchangeFormat.PYTORCH,
        tasks: list[str] = None,
        entry_timeout: float = 30.0,
        wait_send_task_data_all_clients_timeout: float = 300.0,
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
        self.wait_task_data_sent_to_all_clients_timeout = wait_send_task_data_all_clients_timeout
        self.engine: StreamableEngine = None
        self.sender: TensorSender = None
        self.receiver: TensorReceiver = None
        self.start_sending_time: dict[int, float] = defaultdict(float)
        self.seen_tasks: dict[int, set[str]] = defaultdict(set)
        self.num_task_data_sent: dict[int, int] = defaultdict(int)
        self.num_task_skipped: dict[int, int] = defaultdict(int)
        self.data_cleaned: dict[int, bool] = defaultdict(bool)
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
            self.sender = TensorSender(engine, FLContextKey.TASK_DATA, self.format, self.tasks)
        except Exception as e:
            self.system_panic(str(e), fl_ctx)
            return

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Handle events for the TensorSender component.

        Args:
            event_type (str): The type of event to handle.
            fl_ctx (FLContext): The FLContext for the current operation.
        """
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.BEFORE_TASK_DATA_FILTER:
            current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
            task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
            self.seen_tasks[current_round].add(task_id)
        elif event_type == EventType.AFTER_TASK_DATA_FILTER:
            # Store tensors after filtering (to get the filtered reference)
            # Then send to each client
            self.sender.store_tensors(fl_ctx)
            self.send_tensors_to_client(fl_ctx)
            num_clients = len(self.engine.get_clients())
            self.wait_sending_task_data_all_clients(num_clients, fl_ctx)
            self.try_to_clean_task_data(num_clients, fl_ctx)
        elif event_type == EventType.BEFORE_TASK_RESULT_FILTER:
            task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
            peer_name = fl_ctx.get_peer_context().get_identity_name()
            try:
                self.receiver.wait_for_tensors(task_id, peer_name)
                self.receiver.set_ctx_with_tensors(fl_ctx)
            except Exception as e:
                self.system_panic(str(e), fl_ctx)
        elif event_type == AppEventType.ROUND_DONE:
            current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
            # Clear received tensors in case they were set back to the FLContext
            # it can happen when the aggregator only accepts part of the clients
            for task_id in self.seen_tasks[current_round]:
                if task_id in self.receiver.tensors:
                    self.receiver.tensors.pop(task_id)

            self.clean_counters(current_round)

    def clean_counters(self, current_round: int):
        """Clean the counters for the current round.

        Args:
            current_round (int): The current round number.
        """
        with self.lock:
            self.num_task_data_sent.pop(current_round, None)
            self.num_task_skipped.pop(current_round, None)
            self.data_cleaned.pop(current_round, None)
            self.seen_tasks.pop(current_round, None)
            self.start_sending_time.pop(current_round, None)

    def send_tensors_to_client(self, fl_ctx: FLContext):
        """Send tensors to the client after task data filtering.

        Args:
            fl_ctx (FLContext): The FLContext for the current operation.
        """
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        with self.lock:
            if not self.start_sending_time.get(current_round):
                self.start_sending_time[current_round] = time.time()

        try:
            self.sender.send(fl_ctx, self.entry_timeout)
        except ValueError as e:
            self.log_warning(fl_ctx, f"No tensors to send to client: {str(e)}")
            success = False
        else:
            success = True

        with self.lock:
            if success:
                self.num_task_data_sent[current_round] += 1
            else:
                self.num_task_skipped[current_round] += 1

    def wait_sending_task_data_all_clients(self, num_clients: int, fl_ctx: FLContext):
        """Wait until all clients have received the task data tensors or timeout occurs.

        Args:
            num_clients (int): The number of clients to wait for.
            fl_ctx (FLContext): The FLContext for the current operation.

        Raises:
            TimeoutError: If not all clients have received the tensors within the timeout period.
        """
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        wait_timeout = self.wait_task_data_sent_to_all_clients_timeout
        while True:
            time.sleep(0.1)

            num_processed = self.num_task_data_sent[current_round] + self.num_task_skipped[current_round]
            if num_processed >= num_clients:
                return

            if time.time() - self.start_sending_time[current_round] > wait_timeout:
                self.system_panic(
                    "Timeout waiting for all clients to receive tensors. "
                    f"Sent to {self.num_task_data_sent[current_round]} out of {num_clients},"
                    f" skipped {self.num_task_skipped[current_round]}.",
                    fl_ctx,
                )
                return

    def try_to_clean_task_data(self, num_clients: int, fl_ctx: FLContext):
        """Clean the task data in the FLContext.

        Args:
            num_clients (int): The number of clients to wait for.
            fl_ctx (FLContext): The FLContext to clean the task data from.
        """
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        # only clean if we successfully sent to all clients
        with self.lock:
            if not self.data_cleaned[current_round] and self.num_task_data_sent[current_round] >= num_clients:
                self.log_info(
                    fl_ctx,
                    f"Tensors were sent to all clients, removing them from task data. "
                    f"Sent {self.num_task_data_sent[current_round]} out of {num_clients}",
                )
                clean_task_data(fl_ctx)
                self.data_cleaned[current_round] = True

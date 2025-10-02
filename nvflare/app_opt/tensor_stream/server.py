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
        root_keys (list[str]): The root keys to include in the tensor sending. Default is None, which means all keys.
        entry_timeout (float): Timeout for tensor entry transfer operations. Default is 30.0 seconds.
        engine (StreamableEngine): The StreamableEngine used for tensor streaming.
        sender (TensorSender): The TensorSender used to send tensors to clients.
        receiver (TensorReceiver): The TensorReceiver used to receive tensors from clients.
        num_task_data_sent (int): The number of task data sent to clients.
        data_cleaned (bool): Flag indicating whether the task data has been cleaned from the FLContext.
    Methods:
        initialize(fl_ctx): Initializes the TensorServerStreamer component.
        handle_event(event_type, fl_ctx): Handles events for the TensorSender component.
        send_tensors_to_client(fl_ctx): Sends tensors to the client after task data filtering.
        try_to_clean_task_data(fl_ctx): Cleans the task data in the FLContext if all clients have received the tensors.
    """

    def __init__(
        self, format: ExchangeFormat = ExchangeFormat.PYTORCH, root_keys: list[str] = None, entry_timeout=30.0
    ):
        """Initialize the TensorServerStreamer component.

        Args:
            format (ExchangeFormat): The format of the tensors to send/receive. Default is ExchangeFormat.TORCH.
            root_keys (list[str]): The root keys to include in the tensor sending. Default is None, which means all keys.
            entry_timeout (float): Timeout for tensor entry transfer operations. Default is 30.0 seconds.
        """
        super().__init__()
        self.format = format
        self.root_keys = root_keys if root_keys is not None else [""]
        self.entry_timeout = entry_timeout
        self.engine: StreamableEngine = None
        self.sender: TensorSender = None
        self.receiver: TensorReceiver = None
        self.num_task_data_sent = 0
        self.data_cleaned = False

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

        self.sender = TensorSender(engine, FLContextKey.TASK_DATA, self.root_keys)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Handle events for the TensorSender component.

        Args:
            event_type (str): The type of event to handle.
            fl_ctx (FLContext): The FLContext for the current operation.
        """
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.BEFORE_TASK_DATA_FILTER:
            self.data_cleaned = False
        elif event_type == EventType.AFTER_TASK_DATA_FILTER:
            try:
                self.send_tensors_to_client(fl_ctx)
            except Exception as e:
                self.system_panic(f"Failed to send tensors: {e}", fl_ctx)
                return

            self.try_to_clean_task_data(fl_ctx)

        elif event_type == EventType.BEFORE_TASK_RESULT_FILTER:
            self.receiver.set_ctx_with_tensors(fl_ctx)

    def send_tensors_to_client(self, fl_ctx: FLContext):
        """Send tensors to the client after task data filtering.

        Args:
            fl_ctx (FLContext): The FLContext for the current operation.
        """
        try:
            self.sender.send(fl_ctx, self.entry_timeout)
        except ValueError as e:
            self.system_panic(f"Failed to send tensors: {e}", fl_ctx)
            return

        self.num_task_data_sent += 1

    def try_to_clean_task_data(self, fl_ctx: FLContext):
        """Clean the task data in the FLContext.

        Args:
            fl_ctx (FLContext): The FLContext to clean the task data from.
        """
        total_clients = len(self.engine.get_clients())
        if self.num_task_data_sent < total_clients:
            self.log_debug(
                fl_ctx,
                f"Not all sites received the tensors yet. "
                f"Skipping removing tensors from task data. Sent {self.num_task_data_sent} out of {total_clients}",
            )

            while not self.data_cleaned:
                # must wait until the tensors where sent to all clients
                # and the tensors are cleaned from fl_ctx before releasing the task
                time.sleep(0.1)

            return

        self.log_info(
            fl_ctx,
            f"All tensors sent now. Removing tensors from task data. "
            f"Sent {self.num_task_data_sent} out of {total_clients}",
        )
        clean_task_data(fl_ctx)
        self.data_cleaned = True
        self.num_task_data_sent = 0

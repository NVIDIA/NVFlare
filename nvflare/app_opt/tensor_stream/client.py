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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.streaming import StreamableEngine
from nvflare.client.config import ExchangeFormat

from .receiver import TensorReceiver
from .sender import TensorSender
from .utils import clean_task_result


class TensorClientStreamer(FLComponent):
    """TensorClientSender handles receiving task data and sending task results from/to server.

    It uses a StreamableEngine, TensorReceiver, and TensorSender to manage tensor streaming on the client side.
    Attributes:
        format (str): The format of the tensors to send. Default is "pytorch".
        entry_timeout (float): Timeout for tensor entry transfer operations. Default is 30.0 seconds.
        engine (StreamableEngine): The StreamableEngine used for tensor streaming.
        sender (TensorSender): The TensorSender used to send tensors to the server.
        receiver (TensorReceiver): The TensorReceiver used to receive tensors from the server.
    Methods:
        initialize(fl_ctx): Initializes the TensorClientStreamer component.
        handle_event(event_type, fl_ctx): Handles events for the TensorSender component.
        send_tensors_to_server(fl_ctx): Sends tensors to the server before sending the task result.
    """

    def __init__(
        self,
        format: ExchangeFormat = ExchangeFormat.PYTORCH,
        tasks: list[str] = None,
        entry_timeout=30.0,
    ):
        """Initialize the TensorClientStreamer component.

        Args:
            format (ExchangeFormat): The format of the tensors to send. Default is ExchangeFormat.TORCH.
            tasks (list[str]): The list of tasks to send tensors for. Default is None, which means the "train" task.
            entry_timeout (float): Timeout for tensor entry transfer operations. Default is 30.0 seconds.
        """
        super().__init__()
        self.format = format
        self.tasks = tasks if tasks is not None else ["train"]
        self.entry_timeout = entry_timeout
        self.engine: StreamableEngine = None
        self.sender: TensorSender = None
        self.receiver: TensorReceiver = None

    def initialize(self, fl_ctx: FLContext):
        """Initialize the TensorClientStreamer component.
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
            self.receiver = TensorReceiver(engine, FLContextKey.TASK_DATA, self.format)
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
            self.receiver.set_ctx_with_tensors(fl_ctx)
            self.receiver.tensors.clear()  # clear previous received tensors
        elif event_type == EventType.AFTER_TASK_RESULT_FILTER:
            self.sender = TensorSender(self.engine, FLContextKey.TASK_RESULT, self.format, self.tasks)
            try:
                self.send_tensors_to_server(fl_ctx)
            except Exception as e:
                self.system_panic(str(e), fl_ctx)

    def send_tensors_to_server(self, fl_ctx: FLContext):
        """Sends tensors to the server before sending the task result.

        Args:
            fl_ctx (FLContext): The FLContext for the current operation.
        """
        if self.sender.send(fl_ctx, self.entry_timeout):
            clean_task_result(fl_ctx)

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod

from nvflare.apis.dxo import from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_opt.p2p.types import LocalConfig, Neighbor


class BaseDistOptExecutor(Executor, ABC):
    """Base class for peer-to-peer distributed optimization algorithm executors.

    This class provides the foundational structure for executors that run P2P distributed
    optimization algorithms. It handles the configuration of the local network,
    execution flow based on tasks, and interaction with neighboring clients.

    Attributes:
        id (int): Unique identifier for the client.
        client_name (str): Name of the client.
        config (LocalConfig): Local configuration containing neighbor information.
        _weight (float): Weight assigned to the client in the network topology.
        neighbors (list[Neighbor]): List of neighboring clients.
    """

    def __init__(self):
        super().__init__()

        self.id = None
        self.client_name = None
        self.config = None
        self._weight = None

        self.neighbors: list[Neighbor] = []

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ):
        if task_name == "config":
            # Load local network config
            self.config = LocalConfig(**from_shareable(shareable).data)
            self.neighbors = self.config.neighbors
            self._weight = 1.0 - sum([n.weight for n in self.neighbors])
            return make_reply(ReturnCode.OK)

        elif task_name == "run_algorithm":
            # Run the algorithm
            self._pre_algorithm_run(fl_ctx, shareable, abort_signal)
            self.run_algorithm(fl_ctx, shareable, abort_signal)
            self._post_algorithm_run(fl_ctx, shareable, abort_signal)
            return make_reply(ReturnCode.OK)
        else:
            self.log_warning(fl_ctx, f"Unknown task name: {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)

    @abstractmethod
    def run_algorithm(self, fl_ctx: FLContext, shareable: Shareable, abort_signal: Signal):
        """Abstract method to execute the main P2P algorithm.

        Subclasses must implement this method to define the algorithm logic.
        """
        pass

    def _pre_algorithm_run(self, fl_ctx: FLContext, shareable: Shareable, abort_signal: Signal):
        """Hook method executed before running the main algorithm.

        Can be overridden to perform setup tasks.
        """
        pass

    def _post_algorithm_run(self, fl_ctx: FLContext, shareable: Shareable, abort_signal: Signal):
        """Hook method executed after running the main algorithm.

        Can be overridden to perform cleanup tasks or save results.
        """
        pass

    @abstractmethod
    def _exchange_values(self, fl_ctx: FLContext, value: any, *args, **kwargs):
        """Abstract method to handle value exchange with neighbors.

        Subclasses must implement this method to define how values are exchanged.

        Args:
            fl_ctx (FLContext): Federated learning context.
            value (any): The value to exchange.

        """
        pass

    @abstractmethod
    def _handle_neighbor_value(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """Abstract method to process incoming values from neighbors.

        Subclasses must implement this method to handle incoming data.

        Args:
            topic (str): Topic of the incoming message.
            request (Shareable): The incoming data from a neighbor.
            fl_ctx (FLContext): Federated learning context.

        Returns:
            Shareable: A response message or data to send back.
        """
        pass

    def _to_message(self, x):
        """Converts a value to a message format suitable for transmission.

        Args:
            x (any): The value to convert.

        Returns:
            any: The converted message.
        """
        return x

    def _from_message(self, x):
        """Converts a received message back to its original value format.

        Args:
            x (any): The received message.

        Returns:
            any: The original value.
        """
        return x

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.client_name = fl_ctx.get_identity_name()
            self.id = int(self.client_name.split("-")[1])

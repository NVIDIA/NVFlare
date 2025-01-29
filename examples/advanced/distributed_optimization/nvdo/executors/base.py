import threading
from abc import abstractmethod
from collections import defaultdict

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

from nvdo.types import LocalConfig, Neighbor


class BaseAlgorithmExecutor(Executor):
    """Base class for algorithm executors."""
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
    def run_algorithm(
        self, fl_ctx: FLContext, shareable: Shareable, abort_signal: Signal
    ):
        """Executes the algorithm"""
        pass

    def _pre_algorithm_run(
        self, fl_ctx: FLContext, shareable: Shareable, abort_signal: Signal
    ):
        """Executes before algorithm run."""
        pass

    def _post_algorithm_run(
        self, fl_ctx: FLContext, shareable: Shareable, abort_signal: Signal
    ):
        """Executes after algorithm run. Could be used, for example, to save results"""
        pass

    def _to_message(self, x):
        return x

    def _from_message(self, x):
        return x
    
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.client_name = fl_ctx.get_identity_name()
            self.id = int(self.client_name.split("-")[1])


class SynchronousAlgorithmExecutor(BaseAlgorithmExecutor):
    """An executor to implement synchronous algorithms."""
    def __init__(self):
        super().__init__()

        self.neighbors_values = defaultdict(dict)

        self.sync_waiter = threading.Event()
        self.lock = threading.Lock()

    def _exchange_values(self, fl_ctx: FLContext, value: any, iteration: int):
        engine = fl_ctx.get_engine()

        # Clear the event before starting the exchange
        self.sync_waiter.clear()

        _ = engine.send_aux_request(
            targets=[neighbor.id for neighbor in self.neighbors],
            topic="send_value",
            request=DXO(
                data_kind=DataKind.WEIGHTS,
                data={
                    "value": self._to_message(value),
                    "iteration": iteration,
                },
            ).to_shareable(),
            timeout=10,
            fl_ctx=fl_ctx,
        )

        # check if neighbors already sent their values
        if len(self.neighbors_values[iteration]) < len(self.neighbors):
            # wait for all neighbors to send their values for the current iteration
            # if not received after timeout, abort the job
            if not self.sync_waiter.wait(timeout=10):
                self.system_panic("failed to receive values from all neighbors", fl_ctx)
                return

    def _handle_neighbor_value(
        self, topic: str, request: Shareable, fl_ctx: FLContext
    ) -> Shareable:
        sender = request.get_peer_props()["__identity_name__"]
        data = from_shareable(request).data
        iteration = data["iteration"]

        with self.lock:
            self.neighbors_values[iteration][sender] = self._from_message(data["value"])
            # Check if all neighbor values have been received
            if len(self.neighbors_values[iteration]) >= len(self.neighbors):
                self.sync_waiter.set()  # Signal that we have all neighbor values
        return make_reply(ReturnCode.OK)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        super().handle_event(event_type, fl_ctx)
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()

            engine.register_aux_message_handler(
                topic="send_value", message_handle_func=self._handle_neighbor_value
            )

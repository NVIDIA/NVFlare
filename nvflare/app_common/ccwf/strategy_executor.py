from typing import List, Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_api import Strategy
from nvflare.apis.fl_api.communication.wf_comm_client_layers import ClientCommLayer
from nvflare.apis.fl_api.message.message_type import MessageType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.wf_comm_client import WFCommClient
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.ccwf.common import StrategyConstants


class StrategyExecutor(Executor):

    def __init__(self, strategy: Strategy):
        super().__init__()
        self.strategy = strategy
        self.communicator = None

    def start_executor(self, fl_ctx: FLContext):
        self.communicator = WFCommClient(max_task_timeout=self.strategy.strategy_config.max_task_timeout)
        comm_layer = ClientCommLayer(self.communicator, fl_ctx)
        self.strategy.initialize(comm_layer)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        message = shareable.get(StrategyConstants.INPUT, None)
        if message is None:
            raise RuntimeError("there is no strategy input")
        elif not isinstance(message, MessageType):
            raise RuntimeError(f"expecting FLMessage or MessageEnvelop, but get {type(message)}")
        elif "clients" not in message.meta:
            raise RuntimeError(f"expecting 'clients' key in message.meta, but not found")

        selected_clients = message.meta["clients"]
        result: Optional[MessageType] = self.strategy.coordinate(selected_clients)
        if result is None:
            raise RuntimeError(f"result shouldn't None")

        return Shareable({StrategyConstants.OUTPUT: result})

    def stop_executor(self, fl_ctx: FLContext):
        self.strategy.finalize()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.start_executor(fl_ctx)

        elif event_type == EventType.BEFORE_PULL_TASK:
            # todo
            pass
        elif event_type in [EventType.ABORT_TASK, EventType.END_RUN]:
            # todo
            self.strategy.finalize()
            pass
        elif event_type == EventType.FATAL_SYSTEM_ERROR:
            self.stop_executor(fl_ctx)
            pass





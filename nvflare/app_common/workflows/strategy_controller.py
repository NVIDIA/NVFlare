from typing import List

from nvflare.apis.fl_api import Strategy
from nvflare.apis.fl_api.communication.wf_comm_server_layers import ServerCommLayer
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.signal import Signal


class StrategyController(Controller):

    def __init__(self, strategy: Strategy):
        super().__init__()
        self.strategy = strategy

    def start_controller(self, fl_ctx: FLContext):
        comm = ServerCommLayer(self.communicator, fl_ctx)
        self.strategy.initialize(comm)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        clients: List[str] = [client.name for client in fl_ctx.get_engine().get_clients()]
        # todo: stopping and error handling
        self.strategy.coordinate(clients)

    def stop_controller(self, fl_ctx: FLContext):
        self.strategy.finalize()

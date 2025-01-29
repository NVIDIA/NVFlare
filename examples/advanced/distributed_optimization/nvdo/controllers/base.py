from nvflare.apis.controller_spec import Task
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.signal import Signal

from nvdo.types import Config


class AlgorithmController(Controller):
    """Controller for running a p2p algorithm on a network."""
    def __init__(
        self,
        config: Config,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config = config

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        # Send network config (aka neighors info) to each client
        for node in self.config.network.nodes:
            task = Task(
                name="config",
                data=DXO(
                    data_kind=DataKind.APP_DEFINED,
                    data={"neighbors": [n.__dict__ for n in node.neighbors]},
                ).to_shareable(),
            )
            self.send_and_wait(task=task, targets=[node.id], fl_ctx=fl_ctx)

        # Run algorithm (with extra params if any passed as data)
        targets = [node.id for node in self.config.network.nodes]
        self.broadcast_and_wait(
            task=Task(
                name="run_algorithm",
                data=DXO(
                    data_kind=DataKind.APP_DEFINED,
                    data={key: value for key, value in self.config.extra.items()},
                ).to_shareable(),
            ),
            targets=targets,
            min_responses=0,
            fl_ctx=fl_ctx,
        )

    def start_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "BaseController started")

    def stop_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "BaseController stopped")

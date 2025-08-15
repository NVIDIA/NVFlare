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
from nvflare.apis.controller_spec import Task
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.signal import Signal
from nvflare.app_opt.p2p.types import Config


class DistOptController(Controller):
    """Controller for running a peer-to-peer distributed optimization algorithm on a network.

    This controller manages the execution of a distributed optimization algorithm by configuring
    each client with its neighbors and initiating the algorithm execution across the network.

    Args:
        config (Config): The P2P network configuration containing node and neighbor information.
    """

    def __init__(
        self,
        config: Config,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config = config

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        # Send network config (aka neighbors info) to each client
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
        self.log_info(fl_ctx, "P2PAlgorithmController started")

    def stop_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "P2PAlgorithmController stopped")

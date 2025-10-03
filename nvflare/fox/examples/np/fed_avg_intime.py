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
from nvflare.fox.api.app import ServerApp
from nvflare.fox.examples.np.algos.client import NPTrainer
from nvflare.fox.examples.np.algos.filters import AddNoiseToModel, PrintCall, PrintResult
from nvflare.fox.examples.np.algos.strategies import NPFedAvgInTime
from nvflare.fox.examples.np.algos.widgets import MetricReceiver
from nvflare.fox.sim.simulator import Simulator


def main():

    server_app = ServerApp(
        strategy_name="fed_avg_in_time",
        strategy=NPFedAvgInTime(initial_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], num_rounds=2),
    )

    server_app.add_collab_object("metric_receiver", MetricReceiver())
    server_app.add_outgoing_call_filters("*.train", [AddNoiseToModel()])
    server_app.add_incoming_result_filters("*.train", [PrintResult()])

    client_app = NPTrainer(delta=1.0)
    client_app.add_incoming_call_filters("*.train", [PrintCall()])
    client_app.add_outgoing_result_filters("*.train", [PrintResult()])

    simulator = Simulator(
        server_app=server_app,
        client_app=client_app,
        num_clients=2,
    )

    simulator.run()


if __name__ == "__main__":
    main()

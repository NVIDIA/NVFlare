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
import logging

from nvflare.fox.api.utils import simple_logging
from nvflare.fox.examples import get_experiment_root
from nvflare.fox.examples.np.algos.client import NPTrainer
from nvflare.fox.examples.np.algos.filters import AddNoiseToModel, Print
from nvflare.fox.examples.np.algos.strategies.avg_intime import NPFedAvgInTime
from nvflare.fox.examples.np.algos.widgets import MetricReceiver
from nvflare.fox.sim.foxsimulator import FoxSimulator


def main():
    simple_logging(logging.DEBUG)

    simulator = FoxSimulator(
        root_dir=get_experiment_root(),
        experiment_name="fedavg_intime",
        server=NPFedAvgInTime(initial_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], num_rounds=2),
        client=NPTrainer(delta=1.0),
        server_objects={"metric_receiver": MetricReceiver()},
        num_clients=2,
    )

    simulator.add_server_outgoing_call_filters("*.train", [AddNoiseToModel()])
    simulator.add_server_incoming_result_filters("*.train", [Print()])
    simulator.set_server_prop("default_timeout", 8.0)

    simulator.add_client_incoming_call_filters("*.train", [Print()])
    simulator.add_client_outgoing_result_filters("*.train", [Print()])
    simulator.set_client_prop("default_timeout", 5.0)

    result = simulator.run()
    print(f"final model: {result}")


if __name__ == "__main__":
    main()

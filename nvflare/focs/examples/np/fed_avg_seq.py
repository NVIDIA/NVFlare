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
import numpy as np

from nvflare.focs.api.app import ServerApp
from nvflare.focs.examples.np.algos.client import TrainerFactory
from nvflare.focs.examples.np.algos.strategies import NPFedAvgSequential
from nvflare.focs.examples.np.algos.widgets import MetricReceiver
from nvflare.focs.sim.runner import AppRunner


def main():

    server_app = ServerApp(
        strategy=NPFedAvgSequential(
            num_rounds=2,
            initial_model=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        )
    )
    server_app.add_target_object("metric_receiver", MetricReceiver())

    runner = AppRunner(
        server_app=server_app,
        client_app=TrainerFactory(delta=1.0),
        num_clients=2,
    )

    runner.run()


if __name__ == "__main__":
    main()

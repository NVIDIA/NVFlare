# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""One CPU-only round with assertions on both clients' actual returned arrays."""

import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


class SmokeStrategy(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        assert not failures, failures
        assert len(results) == 2
        for _, result in results:
            arrays = parameters_to_ndarrays(result.parameters)
            assert len(arrays) == 1 and np.array_equal(arrays[0], np.ones(1024, dtype=np.float32))
        aggregated = super().aggregate_fit(server_round, results, failures)
        print("FLOWER_RUNTIME_ROUND_OK clients=2", flush=True)
        return aggregated


def server_fn(context):
    return ServerAppComponents(
        strategy=SmokeStrategy(
            min_available_clients=2,
            min_fit_clients=2,
            fraction_evaluate=0.0,
            initial_parameters=ndarrays_to_parameters([np.zeros(1024, dtype=np.float32)]),
        ),
        config=ServerConfig(num_rounds=1),
    )


class SmokeClient(NumPyClient):
    def fit(self, parameters, config):
        return [parameters[0] + np.float32(1)], 1, {}


def client_fn(context):
    return SmokeClient().to_client()


server_app = ServerApp(server_fn=server_fn)
client_app = ClientApp(client_fn=client_fn)

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
import networkx as nx
import numpy as np

from nvflare.app_opt.p2p.types import Neighbor, Network, Node
from nvflare.app_opt.p2p.utils.topology import doubly_stochastic_adjacency


def generate_random_network(
    num_clients: int,
    seed: int = 42,
    connection_probability: float = 0.3,
) -> Network:
    """Generate a random configuration for the given number of clients.
    The configuration includes the number of iterations, the network topology,
    and the initial values for each node.

    Args:
        num_clients (int): The number of clients in the network.

    Returns:
        BaseConfig: The generated configuration.
        np.ndarray: The weighted adjacency matrix of the network.
    """
    np.random.seed(seed=seed)

    while True:
        graph = nx.gnp_random_graph(num_clients, p=connection_probability)
        if nx.is_connected(graph):
            break
    adjacency_matrix = nx.adjacency_matrix(graph) + np.eye(num_clients)
    weighted_adjacency_matrix = doubly_stochastic_adjacency(graph)

    network = []
    for j in range(num_clients):
        in_neighbors = np.nonzero(adjacency_matrix[:, j])[0].tolist()
        in_weights = weighted_adjacency_matrix[:, j].tolist()

        neighbors = [Neighbor(id=f"site-{i + 1}", weight=in_weights[i]) for i in in_neighbors if i != j]

        network.append(
            Node(
                id=f"site-{j + 1}",
                neighbors=neighbors,
            )
        )

    config = Network(
        nodes=network,
    )
    return config, weighted_adjacency_matrix

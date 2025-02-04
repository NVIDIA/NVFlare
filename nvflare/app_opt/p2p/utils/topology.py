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
import torch


def doubly_stochastic_adjacency(graph: nx.Graph) -> torch.Tensor:
    """Using Metropolis-Hastings algorithm to compute a doubly stochastic adjacency matrix."""
    num_agents = len(graph.nodes())
    binary_adjacency_matrix = torch.from_numpy(nx.to_numpy_array(graph)).float()
    degree = torch.sum(binary_adjacency_matrix, dim=0)
    W = torch.zeros((num_agents, num_agents))
    for i in range(num_agents):
        N_i = torch.nonzero(binary_adjacency_matrix[i, :])
        for j in N_i:
            W[i, j] = 1 / (1 + max(degree[i], degree[j]))
        W[i, i] = 1 - torch.sum(W[i, :])
    return W


def get_matrix_steady_state(A: torch.Tensor):
    """Get the steady state of a matrix via eigendecomposition"""
    eigenvalues, eigenvectors = torch.linalg.eig(A)  # unnormalized and unordered eigenvalues and eigenvectors
    idx = eigenvalues.float().argsort(descending=True)
    eigenvectors = eigenvectors.float()[:, idx]

    steady_state = eigenvectors[:, 0] / torch.sum(eigenvectors[:, 0])
    return steady_state

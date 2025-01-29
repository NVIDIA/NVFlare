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
    eigenvalues, eigenvectors = torch.linalg.eig(
        A
    )  # unnormalized and unordered eigenvalues and eigenvectors
    idx = eigenvalues.float().argsort(descending=True)
    eigenvectors = eigenvectors.float()[:, idx]

    steady_state = eigenvectors[:, 0] / torch.sum(eigenvectors[:, 0])
    return steady_state

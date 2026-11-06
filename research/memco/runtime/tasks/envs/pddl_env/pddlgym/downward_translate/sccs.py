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

"""Tarjan's algorithm for maximal strongly connected components.

We provide two versions of the algorithm for different graph
representations.

Since the original recursive version exceeds python's maximal
recursion depth on some planning instances, this is an iterative
version with an explicit recursion stack (iter_stack).

Note that the derived graph where each SCC is a single "supernode" is
necessarily acyclic. The SCCs returned by the algorithm are in a
topological sort order with respect to this derived DAG.
"""

from collections import defaultdict

__all__ = ["get_sccs_adjacency_list", "get_sccs_adjacency_dict"]


def get_sccs_adjacency_list(adjacency_list):
    """Compute SCCs for a graph represented as an adjacency list.

    `adjacency_list` is a list (or similar data structure) whose
    indices correspond to the graph nodes. For example, if
    `len(adjacency_list)` is N, the graph nodes are {0, ..., N-1}.

    For every node `u`, `adjacency_list[u]` is the list (or similar data
    structure) of successors of `u`.

    Returns a list of lists that defines a partition of {0, ..., N-1},
    where each block in the partition is an SCC of the graph, and
    the partition is given in a topologically sort order."""
    return StronglyConnectedComponentComputation(adjacency_list).get_result()


def get_sccs_adjacency_dict(adjacency_dict):
    """Compute SCCs for a graph represented as an adjacency dict.

    `adjacency_dict` is a dictionary whose keys are the vertices of
    the graph.

    For every node `u`, adjacency_dict[u]` is the list (or similar
    data structure) of successors of `u`.

    Returns a list of lists that defines a partition of the graph
    nodes, where each block in the partition is an SCC of the graph,
    and the partition is given in a topologically sort order."""
    node_to_index = {}
    index_to_node = []
    for index, node in enumerate(adjacency_dict):
        node_to_index[node] = index
        index_to_node.append(node)

    adjacency_list = []
    for index, node in enumerate(index_to_node):
        successors = adjacency_dict[node]
        successor_indices = [node_to_index[v] for v in successors]
        adjacency_list.append(successor_indices)

    result_indices = get_sccs_adjacency_list(adjacency_list)

    result = []
    for block_indices in result_indices:
        block = [index_to_node[index] for index in block_indices]
        result.append(block)
    return result


class StronglyConnectedComponentComputation:
    def __init__(self, unweighted_graph):
        self.graph = unweighted_graph
        self.BEGIN, self.CONTINUE, self.RETURN = 0, 1, 2  # "recursion" handling

    def get_result(self):
        self.indices = dict()
        self.lowlinks = defaultdict(lambda: -1)
        self.stack_indices = dict()
        self.current_index = 0
        self.stack = []
        self.sccs = []

        for i in range(len(self.graph)):
            if i not in self.indices:
                self.visit(i)
        self.sccs.reverse()
        return self.sccs

    def visit(self, vertex):
        iter_stack = [(vertex, None, None, self.BEGIN)]
        while iter_stack:
            v, w, succ_index, state = iter_stack.pop()

            if state == self.BEGIN:
                self.current_index += 1
                self.indices[v] = self.current_index
                self.lowlinks[v] = self.current_index
                self.stack_indices[v] = len(self.stack)
                self.stack.append(v)

                iter_stack.append((v, None, 0, self.CONTINUE))
            elif state == self.CONTINUE:
                successors = self.graph[v]
                if succ_index == len(successors):
                    if self.lowlinks[v] == self.indices[v]:
                        stack_index = self.stack_indices[v]
                        scc = self.stack[stack_index:]
                        del self.stack[stack_index:]
                        for n in scc:
                            del self.stack_indices[n]
                        self.sccs.append(scc)
                else:
                    w = successors[succ_index]
                    if w not in self.indices:
                        iter_stack.append((v, w, succ_index, self.RETURN))
                        iter_stack.append((w, None, None, self.BEGIN))
                    else:
                        if w in self.stack_indices:
                            self.lowlinks[v] = min(self.lowlinks[v], self.indices[w])
                        iter_stack.append((v, None, succ_index + 1, self.CONTINUE))
            elif state == self.RETURN:
                self.lowlinks[v] = min(self.lowlinks[v], self.lowlinks[w])
                iter_stack.append((v, None, succ_index + 1, self.CONTINUE))

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
"""Scheduler-neutral node-group contract for multi-node training.

Multi-node-capable job launchers export this contract to every node of a
launcher-owned node group. Framework helper commands (torchrun, trackers,
coordinators) read it back with :meth:`NodeGroup.from_env` and translate it
into framework-specific rendezvous arguments. The contract is the minimal
"single-coordinator rendezvous" set: group size, this node's rank, and one
well-known endpoint on node rank 0.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

ENV_NNODES = "NVFL_NNODES"
ENV_NODE_RANK = "NVFL_NODE_RANK"
ENV_MASTER_ADDR = "NVFL_MASTER_ADDR"
ENV_MASTER_PORT = "NVFL_MASTER_PORT"
ENV_RUN_ID = "NVFL_RUN_ID"

CONTRACT_ENV_NAMES = (ENV_NNODES, ENV_NODE_RANK, ENV_MASTER_ADDR, ENV_MASTER_PORT, ENV_RUN_ID)

# Job-spec vocabulary shared by every launcher that adopts the contract: a
# launcher-mode block requests a node group with JOB_SPEC_NODES > 1, and
# JOB_SPEC_NODE_COMMAND is the worker command for the non-zero node ranks.
JOB_SPEC_NODES = "nodes"
JOB_SPEC_NODE_COMMAND = "node_command"

# Launcher modes that implement node groups; adopting launchers add themselves.
NODE_GROUP_MODES = ("slurm",)

DEFAULT_MASTER_PORT = 29400


class NodeGroupError(RuntimeError):
    """A deterministic node-group configuration failure."""


def is_multi_node_env(environ) -> bool:
    """Cheap multi-node predicate matching from_env semantics; garbage counts as absent."""
    value = environ.get(ENV_NNODES)
    return isinstance(value, str) and value.isascii() and value.isdigit() and int(value) > 1


def _positive_int(value, name: str, default: int) -> int:
    if value is None or value == "":
        return default
    if not isinstance(value, str) or not value.isascii() or not value.isdigit() or int(value) <= 0:
        raise NodeGroupError(f"{name} must be a positive integer")
    return int(value)


def _node_rank(value, nnodes: int) -> int:
    if value is None or value == "":
        return 0
    if not isinstance(value, str) or not value.isascii() or not value.isdigit() or int(value) >= nnodes:
        raise NodeGroupError(f"{ENV_NODE_RANK} must be an integer in 0..{nnodes - 1}")
    return int(value)


@dataclass(frozen=True)
class NodeGroup:
    """The node-group contract of one node, parsed and validated."""

    nnodes: int
    node_rank: int
    master_addr: Optional[str]
    master_port: int
    run_id: Optional[str] = None

    @property
    def is_multi_node(self) -> bool:
        return self.nnodes > 1

    @classmethod
    def from_env(cls, environ) -> "NodeGroup":
        """Parse the contract from an environment mapping.

        Absent variables mean a plain single-node run: ``nnodes=1``, rank 0.
        A multi-node group requires a coordinator address.
        """
        nnodes = _positive_int(environ.get(ENV_NNODES), ENV_NNODES, 1)
        node_rank = _node_rank(environ.get(ENV_NODE_RANK), nnodes)
        master_addr = environ.get(ENV_MASTER_ADDR) or None
        if nnodes > 1 and not master_addr:
            raise NodeGroupError(f"{ENV_MASTER_ADDR} must be set for a multi-node group")
        master_port = _positive_int(environ.get(ENV_MASTER_PORT), ENV_MASTER_PORT, DEFAULT_MASTER_PORT)
        if master_port > 65535:
            raise NodeGroupError(f"{ENV_MASTER_PORT} must be at most 65535")
        return cls(
            nnodes=nnodes,
            node_rank=node_rank,
            master_addr=master_addr,
            master_port=master_port,
            run_id=environ.get(ENV_RUN_ID) or None,
        )


def split_training_argv(argv: Sequence[str]) -> tuple:
    """Split helper options from the training command at the ``--`` boundary."""
    try:
        boundary = list(argv).index("--")
    except ValueError as e:
        raise NodeGroupError("'--' is required before the training script") from e
    training_argv = tuple(argv[boundary + 1 :])
    if not training_argv or not training_argv[0]:
        raise NodeGroupError("training script must be specified after '--'")
    return tuple(argv[:boundary]), training_argv

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
"""Run torchrun for one node of an NVFlare node group.

The launcher exports the scheduler-neutral node-group contract (NVFL_NNODES,
NVFL_NODE_RANK, NVFL_MASTER_ADDR, NVFL_MASTER_PORT) to every node. This helper
translates it into torchrun rendezvous arguments, so the same command line works
as the rank-0 training command and as the multi-node ``node_command``:

    python -m nvflare.app_opt.pt.torchrun_node --nproc-per-node=8 -- custom/client.py --epochs 2

Without the contract in the environment it degrades to standalone single-node
torchrun, so one command also covers plain single-node runs.
"""

import argparse
import os
import sys
from typing import Optional, Sequence

ENV_NNODES = "NVFL_NNODES"
ENV_NODE_RANK = "NVFL_NODE_RANK"
ENV_MASTER_ADDR = "NVFL_MASTER_ADDR"
ENV_MASTER_PORT = "NVFL_MASTER_PORT"

_DEFAULT_MASTER_PORT = 29400
_DEFAULT_JOIN_TIMEOUT = 600


class TorchrunNodeError(RuntimeError):
    """A deterministic helper configuration failure."""


def _positive_int(value: Optional[str], name: str, default: int) -> int:
    if value is None or value == "":
        return default
    if not value.isascii() or not value.isdigit() or int(value) <= 0:
        raise TorchrunNodeError(f"{name} must be a positive integer")
    return int(value)


def _node_rank(environ: dict, nnodes: int) -> int:
    value = environ.get(ENV_NODE_RANK)
    if value is None or value == "":
        return 0
    if not value.isascii() or not value.isdigit() or int(value) >= nnodes:
        raise TorchrunNodeError(f"{ENV_NODE_RANK} must be an integer in 0..{nnodes - 1}")
    return int(value)


def _parse_nproc_per_node(value: str) -> str:
    if value == "auto" or (value.isascii() and value.isdigit() and int(value) > 0):
        return value
    raise argparse.ArgumentTypeError("must be 'auto' or a positive integer")


def _split_training_argv(argv: Sequence[str]) -> tuple:
    try:
        boundary = list(argv).index("--")
    except ValueError as e:
        raise TorchrunNodeError("'--' is required before the training script") from e
    training_argv = tuple(argv[boundary + 1 :])
    if not training_argv or not training_argv[0]:
        raise TorchrunNodeError("training script must be specified after '--'")
    return tuple(argv[:boundary]), training_argv


def build_torchrun_argv(argv: Sequence[str], environ: dict) -> list:
    option_argv, training_argv = _split_training_argv(argv)
    parser = argparse.ArgumentParser(prog=f"{sys.executable} -m {__spec__.name if __spec__ else __name__}")
    parser.add_argument("--nproc-per-node", type=_parse_nproc_per_node, default="auto")
    parser.add_argument("--join-timeout", type=int, default=_DEFAULT_JOIN_TIMEOUT)
    options = parser.parse_args(option_argv)
    if options.join_timeout <= 0:
        raise TorchrunNodeError("--join-timeout must be a positive integer")

    nnodes = _positive_int(environ.get(ENV_NNODES), ENV_NNODES, 1)
    result = [sys.executable, "-u", "-m", "torch.distributed.run", f"--nproc_per_node={options.nproc_per_node}"]
    if nnodes == 1:
        result.append("--standalone")
    else:
        node_rank = _node_rank(environ, nnodes)
        master_addr = environ.get(ENV_MASTER_ADDR)
        if not master_addr:
            raise TorchrunNodeError(f"{ENV_MASTER_ADDR} must be set for a multi-node group")
        master_port = _positive_int(environ.get(ENV_MASTER_PORT), ENV_MASTER_PORT, _DEFAULT_MASTER_PORT)
        rdzv_id = environ.get("SLURM_JOB_ID") or "nvflare"
        result.extend(
            [
                f"--nnodes={nnodes}",
                f"--node_rank={node_rank}",
                "--rdzv_backend=c10d",
                f"--rdzv_endpoint={master_addr}:{master_port}",
                f"--rdzv_id={rdzv_id}",
                f"--rdzv_conf=join_timeout={options.join_timeout}",
            ]
        )
    result.extend(training_argv)
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    try:
        torchrun_argv = build_torchrun_argv(sys.argv[1:] if argv is None else argv, dict(os.environ))
    except TorchrunNodeError as e:
        print(f"torchrun_node: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    os.execv(sys.executable, torchrun_argv)


if __name__ == "__main__":
    main()

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

This translates NVFlare's node-group environment into torchrun rendezvous
arguments, so the same command line works as the rank-0 training command and
as the command for the other nodes:

    python -m nvflare.app_opt.pt.torchrun_node --nproc-per-node=8 custom/client.py --epochs 2

Without the contract in the environment it degrades to standalone single-node
torchrun, so one command also covers plain single-node runs.
"""

import argparse
import os
import sys
from typing import Sequence

_DEFAULT_JOIN_TIMEOUT = 600
_DEFAULT_MASTER_PORT = 29400

_ENV_NNODES = "NVFL_NNODES"
_ENV_NODE_RANK = "NVFL_NODE_RANK"
_ENV_MASTER_ADDR = "NVFL_MASTER_ADDR"
_ENV_MASTER_PORT = "NVFL_MASTER_PORT"
_ENV_RUN_ID = "NVFL_RUN_ID"


def build_torchrun_argv(argv: Sequence[str], environ: dict) -> list:
    parser = argparse.ArgumentParser(prog=f"{sys.executable} -m {__spec__.name if __spec__ else __name__}")
    parser.add_argument("--nproc-per-node", default="auto")
    parser.add_argument("--join-timeout", type=int, default=_DEFAULT_JOIN_TIMEOUT)
    parser.add_argument("training_argv", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    if options.join_timeout <= 0:
        raise ValueError("--join-timeout must be a positive integer")
    if not options.training_argv:
        raise ValueError("training script must be specified directly after torchrun_node options")

    nnodes = int(environ.get(_ENV_NNODES, 1))
    node_rank = int(environ.get(_ENV_NODE_RANK, 0))
    if nnodes < 1 or not 0 <= node_rank < nnodes:
        raise ValueError("invalid node-group topology")

    result = [sys.executable, "-u", "-m", "torch.distributed.run", f"--nproc_per_node={options.nproc_per_node}"]
    if nnodes == 1:
        result.append("--standalone")
    else:
        master_addr = environ.get(_ENV_MASTER_ADDR)
        if not master_addr:
            raise ValueError(f"{_ENV_MASTER_ADDR} must be set for a multi-node group")
        master_port = environ.get(_ENV_MASTER_PORT, _DEFAULT_MASTER_PORT)
        run_id = environ.get(_ENV_RUN_ID)
        if not run_id:
            raise ValueError(f"{_ENV_RUN_ID} must be set for a multi-node group")
        result.extend(
            [
                f"--nnodes={nnodes}",
                f"--node_rank={node_rank}",
                # Elastic c10d assigns node ranks dynamically and ignores
                # --node_rank. Static rendezvous keeps the CJ node at global rank 0.
                "--rdzv_backend=static",
                f"--rdzv_endpoint={master_addr}:{master_port}",
                f"--rdzv_id={run_id}",
                f"--rdzv_conf=timeout={options.join_timeout}",
            ]
        )
    result.extend(options.training_argv)
    return result


def main() -> None:
    try:
        torchrun_argv = build_torchrun_argv(sys.argv[1:], os.environ)
    except ValueError as e:
        print(f"torchrun_node: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    os.execv(sys.executable, torchrun_argv)


if __name__ == "__main__":
    main()

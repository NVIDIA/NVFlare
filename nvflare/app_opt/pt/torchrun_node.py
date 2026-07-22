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

PyTorch consumer of the scheduler-neutral node-group contract in
:mod:`nvflare.app_common.multinode`. It translates the contract into torchrun
rendezvous arguments, so the same command line works as the rank-0 training
command and as the multi-node ``node_command``:

    python -m nvflare.app_opt.pt.torchrun_node --nproc-per-node=8 -- custom/client.py --epochs 2

Without the contract in the environment it degrades to standalone single-node
torchrun, so one command also covers plain single-node runs.
"""

import argparse
import os
import sys
from typing import Optional, Sequence

from nvflare.app_common.multinode import NodeGroup, NodeGroupError, split_training_argv

_DEFAULT_JOIN_TIMEOUT = 600


class TorchrunNodeError(NodeGroupError):
    """A deterministic helper configuration failure."""


def _parse_nproc_per_node(value: str) -> str:
    if value == "auto" or (value.isascii() and value.isdigit() and int(value) > 0):
        return value
    raise argparse.ArgumentTypeError("must be 'auto' or a positive integer")


def build_torchrun_argv(argv: Sequence[str], environ: dict) -> list:
    option_argv, training_argv = split_training_argv(argv)
    parser = argparse.ArgumentParser(prog=f"{sys.executable} -m {__spec__.name if __spec__ else __name__}")
    parser.add_argument("--nproc-per-node", type=_parse_nproc_per_node, default="auto")
    parser.add_argument("--join-timeout", type=int, default=_DEFAULT_JOIN_TIMEOUT)
    options = parser.parse_args(option_argv)
    if options.join_timeout <= 0:
        raise TorchrunNodeError("--join-timeout must be a positive integer")

    group = NodeGroup.from_env(environ)
    result = [sys.executable, "-u", "-m", "torch.distributed.run", f"--nproc_per_node={options.nproc_per_node}"]
    if not group.is_multi_node:
        result.append("--standalone")
    else:
        rdzv_id = environ.get("SLURM_JOB_ID") or "nvflare"
        result.extend(
            [
                f"--nnodes={group.nnodes}",
                f"--node_rank={group.node_rank}",
                "--rdzv_backend=c10d",
                f"--rdzv_endpoint={group.master_addr}:{group.master_port}",
                f"--rdzv_id={rdzv_id}",
                f"--rdzv_conf=join_timeout={options.join_timeout}",
            ]
        )
    result.extend(training_argv)
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    try:
        torchrun_argv = build_torchrun_argv(sys.argv[1:] if argv is None else argv, dict(os.environ))
    except NodeGroupError as e:
        print(f"torchrun_node: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    os.execv(sys.executable, torchrun_argv)


if __name__ == "__main__":
    main()

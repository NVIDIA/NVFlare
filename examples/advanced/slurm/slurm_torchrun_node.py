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

"""Launch one static multi-node torchrun worker group in a Slurm allocation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, NoReturn, Optional, Sequence

_SCRIPT_PATH = os.path.realpath(__file__)
_PRIVATE_NODE_MODE = "__nvflare_slurm_node__"

_ENV_SRUN = "NVFLARE_SLURM_SRUN"
_ENV_SCONTROL = "NVFLARE_SLURM_SCONTROL"
_ENV_NNODES = "NVFLARE_SLURM_HELPER_NNODES"
_ENV_MASTER_ADDR = "NVFLARE_SLURM_HELPER_MASTER_ADDR"
_ENV_MASTER_PORT = "NVFLARE_SLURM_HELPER_MASTER_PORT"
_ENV_NPROC = "NVFLARE_SLURM_HELPER_NPROC_PER_NODE"

_DEFAULT_PORT_BASE = 29400
_DEFAULT_PORT_SPAN = 1000


class SlurmTorchrunError(RuntimeError):
    """A deterministic helper configuration or launch failure."""


@dataclass(frozen=True)
class _PublicOptions:
    nproc_per_node: str | int
    rdzv_port_base: int
    rdzv_port_span: int
    training_argv: tuple[str, ...]


def _positive_int(value: Optional[str]) -> Optional[int]:
    if not value or not value.isascii() or not value.isdigit():
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _required_positive_int(environ: dict[str, str], name: str) -> int:
    value = _positive_int(environ.get(name))
    if value is None:
        raise SlurmTorchrunError(f"{name} must be a positive integer")
    return value


def _required_numeric_job_id(environ: dict[str, str]) -> int:
    value = environ.get("SLURM_JOB_ID")
    if not value or not value.isascii() or not value.isdigit():
        raise SlurmTorchrunError("SLURM_JOB_ID must be numeric")
    return int(value)


def _required_node_rank(environ: dict[str, str], nnodes: int) -> int:
    value = environ.get("SLURM_NODEID")
    if not value or not value.isascii() or not value.isdigit():
        raise SlurmTorchrunError(f"SLURM_NODEID must be an integer in 0..{nnodes - 1}")
    node_rank = int(value)
    if node_rank >= nnodes:
        raise SlurmTorchrunError(f"SLURM_NODEID must be an integer in 0..{nnodes - 1}")
    return node_rank


def _required_nonempty(environ: dict[str, str], name: str) -> str:
    value = environ.get(name)
    if not value:
        raise SlurmTorchrunError(f"{name} must be set")
    return value


def _command(environ: dict[str, str], name: str, default: str) -> str:
    return environ.get(name) or default


def _parse_nproc_per_node(value: str) -> str | int:
    if value == "auto":
        return value
    parsed = _positive_int(value)
    if parsed is None:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer")
    return parsed


def _training_argv_after_boundary(argv: Sequence[str], mode: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        boundary = argv.index("--")
    except ValueError as e:
        raise SlurmTorchrunError(f"{mode} mode requires '--' before the training script") from e

    before = tuple(argv[:boundary])
    training_argv = tuple(argv[boundary + 1 :])
    if not training_argv or not training_argv[0]:
        raise SlurmTorchrunError("training script must be specified after '--'")
    return before, training_argv


def _parse_public_cli(argv: Sequence[str]) -> _PublicOptions:
    option_argv, training_argv = _training_argv_after_boundary(argv, "public")
    parser = argparse.ArgumentParser(prog=f"python {_SCRIPT_PATH}", allow_abbrev=False)
    parser.add_argument("--nproc-per-node", type=_parse_nproc_per_node, default="auto")
    parser.add_argument("--rdzv-port-base", type=int, default=_DEFAULT_PORT_BASE)
    parser.add_argument("--rdzv-port-span", type=int, default=_DEFAULT_PORT_SPAN)
    options = parser.parse_args(option_argv)
    return _PublicOptions(
        nproc_per_node=options.nproc_per_node,
        rdzv_port_base=options.rdzv_port_base,
        rdzv_port_span=options.rdzv_port_span,
        training_argv=training_argv,
    )


def _parse_node_cli(argv: Sequence[str]) -> tuple[str, ...]:
    option_argv, training_argv = _training_argv_after_boundary(argv, "private node")
    if option_argv:
        raise SlurmTorchrunError("private node mode does not accept command-line options")
    return training_argv


def _compute_master_port(job_id: int, base: int, span: int) -> int:
    if base < 1024:
        raise SlurmTorchrunError("rdzv port base must be at least 1024")
    if span <= 0:
        raise SlurmTorchrunError("rdzv port span must be positive")
    if base + span - 1 > 65535:
        raise SlurmTorchrunError("rdzv port range must end at or below 65535")
    return base + (job_id % span)


def _discover_hosts(scontrol: str, nodelist: str, nnodes: int) -> tuple[str, ...]:
    result = subprocess.run(
        [scontrol, "show", "hostnames", nodelist],
        shell=False,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise SlurmTorchrunError(f"scontrol show hostnames failed with exit code {result.returncode}{suffix}")

    hosts = tuple(line.strip() for line in result.stdout.splitlines())
    if len(hosts) != nnodes or any(not host for host in hosts) or len(set(hosts)) != nnodes:
        raise SlurmTorchrunError(f"scontrol must return exactly {nnodes} unique nonempty host names")
    return hosts


def _cuda_device_count() -> int:
    try:
        import torch
    except ImportError as e:
        raise SlurmTorchrunError("PyTorch is required when --nproc-per-node=auto cannot use SLURM_GPUS_ON_NODE") from e

    return torch.cuda.device_count()


def _resolve_nproc_per_node(
    requested: str | int,
    environ: dict[str, str],
    cuda_device_count: Callable[[], int] = _cuda_device_count,
) -> int:
    if isinstance(requested, int):
        if isinstance(requested, bool) or requested <= 0:
            raise SlurmTorchrunError("nproc_per_node must be positive")
        return requested
    if requested != "auto":
        raise SlurmTorchrunError("nproc_per_node must be 'auto' or a positive integer")

    slurm_gpu_count = _positive_int(environ.get("SLURM_GPUS_ON_NODE"))
    if slurm_gpu_count is not None:
        return slurm_gpu_count
    detected_gpu_count = cuda_device_count()
    if isinstance(detected_gpu_count, int) and not isinstance(detected_gpu_count, bool) and detected_gpu_count > 0:
        return detected_gpu_count
    return 1


def _build_srun_argv(
    srun: str,
    python: str,
    nnodes: int,
    cpus_per_task: Optional[int],
    training_argv: Sequence[str],
) -> list[str]:
    argv = [
        srun,
        f"--nodes={nnodes}",
        f"--ntasks={nnodes}",
        "--ntasks-per-node=1",
        "--distribution=block:block",
    ]
    if cpus_per_task is not None:
        argv.append(f"--cpus-per-task={cpus_per_task}")
    argv.extend(
        [
            "--kill-on-bad-exit=1",
            "--export=ALL",
            python,
            _SCRIPT_PATH,
            _PRIVATE_NODE_MODE,
            "--",
            *training_argv,
        ]
    )
    return argv


def _build_torchrun_argv(
    python: str,
    nnodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
    nproc_per_node: int,
    training_argv: Sequence[str],
) -> list[str]:
    return [
        python,
        "-m",
        "torch.distributed.run",
        f"--nnodes={nnodes}",
        f"--nproc-per-node={nproc_per_node}",
        f"--node-rank={node_rank}",
        f"--master-addr={master_addr}",
        f"--master-port={master_port}",
        "--max-restarts=0",
        "--",
        *training_argv,
    ]


def _run_public(options: _PublicOptions, environ: dict[str, str]) -> int:
    job_id = _required_numeric_job_id(environ)
    nnodes = _required_positive_int(environ, "SLURM_NNODES")
    nodelist = _required_nonempty(environ, "SLURM_JOB_NODELIST")
    current_node = _required_nonempty(environ, "SLURMD_NODENAME")
    srun = _command(environ, _ENV_SRUN, "srun")
    scontrol = _command(environ, _ENV_SCONTROL, "scontrol")

    hosts = _discover_hosts(scontrol, nodelist, nnodes)
    if current_node != hosts[0]:
        raise SlurmTorchrunError(
            f"helper must run on allocation node 0 ('{hosts[0]}'), but SLURMD_NODENAME is '{current_node}'"
        )

    master_port = _compute_master_port(job_id, options.rdzv_port_base, options.rdzv_port_span)
    nproc_per_node = _resolve_nproc_per_node(options.nproc_per_node, environ)
    cpus_per_task = _positive_int(environ.get("SLURM_CPUS_PER_TASK"))
    child_env = dict(environ)
    child_env.update(
        {
            _ENV_NNODES: str(nnodes),
            _ENV_MASTER_ADDR: hosts[0],
            _ENV_MASTER_PORT: str(master_port),
            _ENV_NPROC: str(nproc_per_node),
        }
    )
    srun_argv = _build_srun_argv(srun, sys.executable, nnodes, cpus_per_task, options.training_argv)

    result = subprocess.run(srun_argv, shell=False, env=child_env, check=False)
    return 128 - result.returncode if result.returncode < 0 else result.returncode


def _run_node(training_argv: Sequence[str], environ: dict[str, str]) -> NoReturn:
    nnodes = _required_positive_int(environ, _ENV_NNODES)
    node_rank = _required_node_rank(environ, nnodes)
    master_addr = _required_nonempty(environ, _ENV_MASTER_ADDR)
    master_port = _required_positive_int(environ, _ENV_MASTER_PORT)
    nproc_per_node = _required_positive_int(environ, _ENV_NPROC)
    torchrun_argv = _build_torchrun_argv(
        sys.executable,
        nnodes,
        node_rank,
        master_addr,
        master_port,
        nproc_per_node,
        training_argv,
    )
    os.execv(sys.executable, torchrun_argv)


def main(argv: Optional[Sequence[str]] = None, environ: Optional[dict[str, str]] = None) -> int:
    argv = tuple(sys.argv[1:] if argv is None else argv)
    environ = os.environ if environ is None else environ
    try:
        if argv and argv[0] == _PRIVATE_NODE_MODE:
            return _run_node(_parse_node_cli(argv[1:]), environ)
        return _run_public(_parse_public_cli(argv), environ)
    except (OSError, SlurmTorchrunError) as e:
        print(f"slurm_torchrun_node: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

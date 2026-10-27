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

"""FedJob wiring + simulator/export launch helpers.

Version-sensitive nvflare API (FedJob, ScriptRunner, simulator invocation)
is confined to this module — re-verify here when bumping the nvflare pin.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def client_ids(cfg) -> list[str]:
    return [f"c{i}" for i in range(cfg.data.num_clients)]


def _workspace(cfg) -> str:
    nv = cfg.fl.nvflare
    return str(Path(nv.workspace or (Path(cfg.fl.output_dir) / "nvflare_workspace")).resolve())


def build_job(cfg, config_path: str):
    """FedJob: DecentVLAController on the server, the Client-API script on clients."""
    from nvflare.client.config import ExchangeFormat, TransferType
    from nvflare.job_config.api import FedJob
    from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

    from decent_vla.runtime.nvflare import client_script
    from decent_vla.runtime.nvflare.controller import DecentVLAController

    abs_config = str(Path(config_path).resolve())
    abs_out = str(Path(cfg.fl.output_dir).resolve())
    # The submission cwd travels with the job: nvflare processes run from a
    # workspace cwd, and restoring this makes config-relative paths (e.g.
    # backend.extra.train_config_path) behave as they do under local mode.
    workdir = str(Path.cwd())
    job = FedJob(name=cfg.name or "decent-vla", min_clients=cfg.data.num_clients)
    job.to_server(
        DecentVLAController(config_path=abs_config, output_dir=abs_out, workdir=workdir)
    )
    job.to_clients(
        ScriptRunner(
            script=str(Path(client_script.__file__).resolve()),
            script_args=f"--config {abs_config} --workdir {workdir}",
            launch_external_process=cfg.fl.nvflare.launch_external_process,
            # NUMPY end-to-end + FULL transfer = the Client API applies no
            # framework conversion or diffing; params are exactly what our
            # codec produced (deltas are computed by run_client_round itself).
            framework=FrameworkType.NUMPY,
            server_expected_format=ExchangeFormat.NUMPY,
            params_transfer_type=TransferType.FULL,
        )
    )
    return job


def split_gpu_groups(gpu: str) -> list[str]:
    """Split a simulator GPU map into groups, mirroring nvflare's `split_gpus`.

    Grammar: comma-separated device ids, `[a,b]` binding several ids into ONE
    group. `"0,1"` is two groups; `"[0,1]"` is one group of two cards.

    Reimplemented rather than imported so the check runs at config-load time
    without nvflare installed (the test suite fakes the nvflare namespace).
    Verified equal to `nvflare.private.fed.utils.fed_utils.split_gpus` on
    nvflare 2.8.1 over every map these configs use.
    """
    out, buf, depth = [], [], 0
    for ch in gpu.replace(" ", ""):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth < 0:
                raise ValueError(f"fl.nvflare.gpu={gpu!r}: brackets not paired")
        elif ch == "," and depth == 0:
            out.append("".join(buf)); buf = []
            continue
        else:
            buf.append(ch)
    if depth:
        raise ValueError(f"fl.nvflare.gpu={gpu!r}: brackets not paired")
    out.append("".join(buf))
    if any(not g for g in out):
        raise ValueError(f"fl.nvflare.gpu={gpu!r}: empty GPU group")
    return out


def check_gpu_map(gpu: str | None, n_clients: int) -> list[str]:
    """Validate `fl.nvflare.gpu` against the client count. Returns the groups.

    NVFlare gives each GPU GROUP exactly one client slot at a time: with more
    than one group it hard-sets `threads = 1` per group
    (simulator_runner.py, "each GPU group will run with only 1 thread"), so a
    group holding several clients runs them one at a time and tears the client
    PROCESS down between them. That reloads the policy every round — measured
    at ~30 s of reload per client per round for SmolVLA — and it silently drops the warm
    Adam/scheduler/dataloader state the whole runtime is built to keep.

    So the map is a client->GPU list, ONE ENTRY PER CLIENT: 8 clients on 2 cards
    is "0,1,0,1,0,1,0,1", not "0,1". A single group ("0", or the bracket form
    "[0,1]" for one client owning two cards) is the other legitimate shape and
    keeps whatever `threads` asks for.
    """
    if not gpu:
        return []
    groups = split_gpu_groups(gpu)
    if 1 < len(groups) < n_clients:
        suggested = ",".join(groups[i % len(groups)] for i in range(n_clients))
        raise ValueError(
            f"fl.nvflare.gpu={gpu!r} declares {len(groups)} GPU groups for "
            f"{n_clients} clients. NVFlare runs ONE client per group at a time, "
            f"so {n_clients - len(groups)} client(s) would be swapped in and out "
            f"— reloading the policy and losing warm optimizer state every round. "
            f'Spell one entry per client: gpu: "{suggested}"'
        )
    return groups


def run_simulator(cfg, config_path: str) -> None:
    """Blocking in-process(-tree) run; raises RuntimeError on simulator failure.

    threads defaults to num_clients, but NVFlare overrides it to 1 whenever the
    GPU map names more than one group — residency comes from having one client
    per group (see `check_gpu_map`), not from the thread count. Without a GPU map
    the thread count is what it says: fewer threads make the simulator swap
    client processes between rounds, which drops the warm in-memory client state
    (Adam moments survive only via fl.persist_client_state then).
    """
    nv = cfg.fl.nvflare
    job = build_job(cfg, config_path)
    cids = client_ids(cfg)
    groups = check_gpu_map(nv.gpu, len(cids))
    if groups:
        log.info("nvflare simulator: %d clients over GPU groups %s (one client per group)",
                 len(cids), groups)
    job.simulator_run(
        _workspace(cfg),
        clients=cids,
        threads=nv.threads or len(cids),
        gpu=nv.gpu,
    )


def export_job(cfg, config_path: str) -> Path:
    """Write a standard NVFlare job folder for POC/provisioned submission."""
    nv = cfg.fl.nvflare
    job_dir = Path(nv.job_dir or (Path(cfg.fl.output_dir) / "nvflare_job")).resolve()
    job = build_job(cfg, config_path)
    job.export_job(str(job_dir))
    return job_dir

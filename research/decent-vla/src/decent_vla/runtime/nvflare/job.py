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

import contextlib
import logging
import os
import shutil
import tempfile
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

# Names of the files the job ships: the client script goes in each app's
# custom/ folder, the run config (and the files it names) in config/.
CLIENT_SCRIPT = "client_script.py"
RUN_CONFIG = "decent_vla_run.yaml"


def client_ids(cfg) -> list[str]:
    return [f"c{i}" for i in range(cfg.data.num_clients)]


def _workspace(cfg) -> str:
    nv = cfg.fl.nvflare
    return str(Path(nv.workspace or (Path(cfg.fl.output_dir) / "nvflare_workspace")).resolve())


@contextlib.contextmanager
def staged_job(config_path: str):
    """Stage every file the job ships, by the name it has inside the job, and
    run the block from that folder. Yields (stage folder, config file names).

    The job carries its own run config, the files that config names
    (`client.packaged_refs`, rewritten to bare names) and the client script,
    so a deployed site needs neither the source tree nor the submitter's
    paths. FedJob resolves relative resource names against the cwd both when
    the job is built and when it is exported, so both happen in here.
    """
    from decent_vla.runtime.nvflare import client_script
    from decent_vla.runtime.nvflare.client import packaged_refs

    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    # Where the job folder and the simulator workspace go is the submitter's
    # business; the sites never read it.
    for key in ("job_dir", "workspace"):
        ((raw.get("fl") or {}).get("nvflare") or {}).pop(key, None)
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="decent_vla_job_") as stage:
        config_files = [RUN_CONFIG]
        for key, holder, field in packaged_refs(raw):
            value = holder.get(field)
            if not value:
                continue
            # Relative paths in a run config resolve against the cwd, as in local mode.
            src = Path(cwd, value)
            if not src.is_file():
                raise FileNotFoundError(f"{key}={value!r}: no such file under {cwd}")
            if src.name in (RUN_CONFIG, CLIENT_SCRIPT) or src.name in config_files:
                raise ValueError(f"{key}: {src.name!r} is already a file name in the job")
            shutil.copy(src, Path(stage, src.name))
            holder[field] = src.name
            config_files.append(src.name)
        Path(stage, RUN_CONFIG).write_text(yaml.safe_dump(raw, sort_keys=False))
        shutil.copy(client_script.__file__, Path(stage, CLIENT_SCRIPT))
        os.chdir(stage)
        try:
            yield Path(stage), config_files
        finally:
            os.chdir(cwd)


class JobConfigFiles:
    """FedJob component that puts files in the app's config/ folder.

    Not in custom/: `nvflare job submit` parses and rewrites every .yaml,
    .json and .conf file under custom/, but in config/ only its own
    config_fed_* files.
    """

    def __init__(self, paths: list[str]):
        self.paths = list(paths)

    def add_to_fed_job(self, job, ctx, **kwargs):
        for path in self.paths:
            job.add_file_source(path, None, "config", ctx)


def build_job(cfg, staged: tuple[Path, list[str]], workdir: str | None = None,
              output_dir: str | None = None):
    """FedJob: DecentVLAController on the server, the Client-API script on clients.

    Call inside `staged_job`, with what it yields. A simulator job runs on this
    host, so it is given this host's cwd (`workdir`) and absolute output folder;
    a job exported for deployment gets neither and carries no host paths (sites
    use `client.apply_site_paths`).
    """
    from nvflare.client.config import ExchangeFormat, TransferType
    from nvflare.job_config.api import FedJob
    from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

    from decent_vla.runtime.nvflare.controller import DecentVLAController

    # min_clients is the job's min_sites: FLARE aborts the job once fewer sites
    # are alive, so it follows the round loop's quorum.
    job = FedJob(name=cfg.name or "decent-vla", min_clients=cfg.fl.min_clients or cfg.data.num_clients)
    # Sites run the installed decent_vla package. Without this FedJob copies the
    # sources of the controller's module and its imports into custom/, a partial
    # package beside the installed one.
    job.set_app_packages(["decent_vla"])
    stage, config_files = staged
    configs = [str(stage / name) for name in config_files]
    job.to_server(DecentVLAController(config_path=RUN_CONFIG, output_dir=output_dir, workdir=workdir))
    job.to_server(JobConfigFiles(configs))
    job.to_clients(JobConfigFiles(configs))
    job.to_clients(
        ScriptRunner(
            # A name inside the job: in-process mode finds it in custom/, and
            # external-process mode runs `python3 -u custom/client_script.py`.
            script=CLIENT_SCRIPT,
            script_args=f"--config {RUN_CONFIG}" + (f" --workdir {workdir}" if workdir else ""),
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
    cids = client_ids(cfg)
    groups = check_gpu_map(nv.gpu, len(cids))
    if groups:
        log.info("nvflare simulator: %d clients over GPU groups %s (one client per group)",
                 len(cids), groups)
    workspace = _workspace(cfg)
    workdir, output_dir = os.getcwd(), str(Path(cfg.fl.output_dir).resolve())
    with staged_job(config_path) as staged:
        job = build_job(cfg, staged, workdir=workdir, output_dir=output_dir)
        job.simulator_run(
            workspace,
            clients=cids,
            threads=nv.threads or len(cids),
            gpu=nv.gpu,
        )


def export_job(cfg, config_path: str) -> Path:
    """Write a standard NVFlare job folder for POC/provisioned submission."""
    nv = cfg.fl.nvflare
    job_dir = Path(nv.job_dir or (Path(cfg.fl.output_dir) / "nvflare_job")).resolve()
    if cfg.fl.round_timeout_s is None:
        log.warning(
            "fl.round_timeout_s is not set: a required client that stops answering stalls its "
            "round forever. Set it for a deployment."
        )
    if not Path(cfg.fl.output_dir).is_absolute():
        log.warning(
            "fl.output_dir is relative, so each site writes under the job's own folder: a "
            "resubmitted job gets a new folder and starts again from round 0. Set an absolute "
            "fl.output_dir to resume."
        )
    with staged_job(config_path) as staged:
        job = build_job(cfg, staged)
        job.export_job(str(job_dir))
    return job_dir

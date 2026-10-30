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

"""Export portable FLARE jobs containing VLA workflows and client scripts."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from pathlib import Path

import yaml

# Names of the files the job ships: the client script goes in each app's
# custom/ folder, the run config (and the files it names) in config/.
CLIENT_SCRIPT = "client_script.py"
RUN_CONFIG = "decent_vla_run.yaml"


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
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="decent_vla_job_") as stage:
        config_files = [RUN_CONFIG]
        for key, holder, field in packaged_refs(raw):
            value = holder.get(field)
            if not value:
                continue
            # Resolve input files against the exporting process cwd.
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


def build_job(cfg, staged: tuple[Path, list[str]]):
    """Wire native FedAvg/FedOpt, a LearnablePersistor and Client API trainers."""
    from nvflare.client.config import ExchangeFormat, TransferType
    from nvflare.job_config.api import FedJob
    from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

    from decent_vla.runtime.nvflare.controller import VLAFedAdam, VLAFedAvg
    from decent_vla.runtime.nvflare.persistor import VLAPersistor

    # min_clients is the job's min_sites: FLARE aborts the job once fewer sites
    # are alive, so it follows the round loop's quorum.
    job = FedJob(name=cfg.name or "decent-vla", min_clients=cfg.fl.min_clients or cfg.data.num_clients)
    # Sites run the installed decent_vla package. Without this FedJob copies the
    # sources of the controller's module and its imports into custom/, a partial
    # package beside the installed one.
    job.set_app_packages(["decent_vla"])
    stage, config_files = staged
    configs = [str(stage / name) for name in config_files]
    job.to_server(VLAPersistor(config_path=RUN_CONFIG), id="persistor")
    workflow = VLAFedAdam if cfg.fl.strategy.name == "fedadam" else VLAFedAvg
    job.to_server(
        workflow(
            persistor_id="persistor",
            num_clients=cfg.data.num_clients,
            num_rounds=cfg.fl.num_rounds,
            ignore_result_error=None,
        )
    )
    job.to_server(JobConfigFiles(configs))
    job.to_clients(JobConfigFiles(configs))
    job.to_clients(
        ScriptRunner(
            # A name inside the job: in-process mode finds it in custom/, and
            # external-process mode runs `python3 -u custom/client_script.py`.
            script=CLIENT_SCRIPT,
            script_args=f"--config {RUN_CONFIG}",
            launch_external_process=cfg.fl.nvflare.launch_external_process,
            # NUMPY end-to-end + FULL transfer = the Client API applies no
            # framework conversion or diffing; params are exactly the clients'
            # wire arrays (deltas are computed in the Client API training loop).
            framework=FrameworkType.NUMPY,
            server_expected_format=ExchangeFormat.NUMPY,
            params_transfer_type=TransferType.FULL,
        )
    )
    return job


def export_job(cfg, config_path: str, job_dir: str | Path = "jobs") -> Path:
    """Export once; use nvflare simulator/job submit to execute the same job."""
    job_dir = Path(job_dir).resolve()
    with staged_job(config_path) as staged:
        build_job(cfg, staged).export_job(str(job_dir))
    return job_dir / cfg.name

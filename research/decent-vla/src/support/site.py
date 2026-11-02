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

"""Site-local configuration and path helpers shared by FLARE components."""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def resolve_packaged_paths(cfg, config_dir) -> None:
    """Point the packaged file references of a job's run config at `config_dir`."""

    def resolve(value):
        return str(Path(config_dir) / value) if value and not os.path.isabs(value) else value

    extra = cfg.backend.extra or {}
    if extra.get("train_config_path"):
        extra["train_config_path"] = resolve(extra["train_config_path"])
    cfg.data.held_out = resolve(cfg.data.held_out)


def load_job_config(config_path):
    """Load the run config packaged in a job, with its packaged files resolved."""
    from src.configs.loader import load_run_config

    cfg = load_run_config(str(config_path))
    resolve_packaged_paths(cfg, Path(config_path).parent)
    return cfg


def apply_site_paths(cfg, run_dir) -> None:
    """Resolve site data against DECENT_VLA_WORKDIR and outputs against the job."""
    site_workdir = os.environ.get("DECENT_VLA_WORKDIR")
    if site_workdir:
        _chdir(site_workdir, "DECENT_VLA_WORKDIR")
    if not os.path.isabs(cfg.fl.output_dir):
        cfg.fl.output_dir = str(Path(run_dir) / cfg.fl.output_dir)
    log.info("site paths: cwd %s, outputs %s", os.getcwd(), cfg.fl.output_dir)


def _chdir(path: str, what: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{what} {path!r} does not exist on this site")
    os.chdir(path)

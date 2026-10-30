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

"""CLI glue for `fl.mode: nvflare` — dispatched from cli/main.py.

Kept out of cli/main.py so the CLI stays free of nvflare imports unless the
mode is actually requested.
"""

from __future__ import annotations

import time
from pathlib import Path


def run_from_cli(cfg, config_path: str, resume: str | None = None) -> None:
    if resume:
        raise SystemExit(
            "Native FLARE auto-resumes from "
            "<fl.output_dir>/checkpoints/server_round_*.pt (set DECENT_VLA_NO_RESUME=1 "
            "to start fresh instead)."
        )
    try:
        import nvflare  # noqa: F401
    except ImportError as e:  # pragma: no cover - missing required dependency
        raise SystemExit("NVFlare is required: pip install -e '.'") from e

    from decent_vla.runtime.nvflare import job as nvjob

    env = cfg.fl.nvflare.env
    if env == "simulator":
        # Blocking; the native workflow writes all checkpoints/artifacts
        # into cfg.fl.output_dir, so post-run eval in cmd_run works unchanged.
        started = time.time()
        try:
            nvjob.run_simulator(cfg, config_path)
        except RuntimeError as e:
            raise SystemExit(f"nvflare simulator run failed: {e}") from e
        # The simulator subprocess exits 0 even when the server workflow died
        # (verified empirically on 2.8.1: a crashed controller aborts the job
        # but the process still returns success), so a scripted/chained run
        # would silently "succeed". The controller's final act is writing
        # server_round_final.pt — require a FRESH one as the success signal
        # (the resume nothing-to-do path also rewrites it).
        final = Path(cfg.fl.output_dir) / "checkpoints" / "server_round_final.pt"
        if not final.exists() or final.stat().st_mtime < started - 1:
            raise SystemExit(
                f"nvflare simulator exited without writing a fresh {final} — the "
                f"server workflow failed; see the server log under "
                f"{cfg.fl.nvflare.workspace or Path(cfg.fl.output_dir) / 'nvflare_workspace'}"
            )
    elif env == "export":
        job_dir = nvjob.export_job(cfg, config_path)
        print(
            f"nvflare job folder written to {job_dir}\n"
            f"Submit it to a running federation, e.g.:\n"
            f"  nvflare poc prepare -c {' '.join(nvjob.client_ids(cfg))} && nvflare poc start\n"
            f"  nvflare job submit -j {job_dir}/{cfg.name or 'decent-vla'}\n"
            f"The job carries its run config; every site needs decent-vla installed\n"
            f"(pip install -e .) and its own data and model weights. Site names must\n"
            f"map to c0..cN-1 (see fl.nvflare.site_prefix); outputs go under each\n"
            f"site's job folder unless fl.output_dir is absolute."
        )
        raise SystemExit(0)
    else:
        raise SystemExit(f"Unknown fl.nvflare.env: {env!r} (use 'simulator' or 'export')")

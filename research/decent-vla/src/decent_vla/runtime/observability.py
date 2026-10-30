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

"""Optional Weights & Biases logging for the NVFlare / LocalRunner training runs.

``wandb_round_log`` records per-client loss to ONE wandb run per experiment
(not one-per-client): called once per round from the single server /
LocalRunner process, logging ``loss/<cid>`` + ``loss/mean``. Keyed by the
experiment label with ``resume="allow"`` so a run spanning several SLURM walls
keeps appending to the SAME wandb run. Enable with ``WANDB_PROJECT`` (and
``WANDB_MODE=online`` if compute nodes have network, else offline + wandb sync).
"""
from __future__ import annotations

import logging
import os
import re


def _log_dir():
    return os.path.join(os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()), "logs")


_WANDB_RUN = None
_WANDB_OFF = False
_WANDB_CFG: dict = {}


def configure_wandb(cfg) -> None:
    """Adopt the run config's `wandb:` block (schemas.WandbConfig).

    Called once by whichever process owns the round loop — server_session for
    NVFlare, the CLI for LocalRunner — so that experiment
    tracking is declared in the YAML next to the hyperparameters it describes
    instead of in a launcher's env vars. Idempotent; a cfg without a wandb
    block leaves the legacy env-var behaviour untouched.
    """
    global _WANDB_CFG, _WANDB_OFF, _WANDB_RUN
    w = getattr(cfg, "wandb", None)
    if w is None:
        return
    new = {
        "enabled": w.enabled, "project": w.project, "entity": w.entity,
        "name": w.name, "group": w.group, "job_type": w.job_type,
        "mode": w.mode, "tags": list(w.tags or []), "notes": w.notes,
    }
    if new != _WANDB_CFG:
        # Re-target: drop any half-built state so the next log() re-inits.
        _WANDB_CFG, _WANDB_OFF, _WANDB_RUN = new, False, None


def _wandb_settings(label):
    """Effective settings, precedence CONFIG > ENV > default.

    Env (WANDB_PROJECT/NAME) still works for configs that declare no
    `wandb:` block.
    """
    c, env = _WANDB_CFG, os.environ.get
    rid = re.sub(r"[^A-Za-z0-9_.-]", "-", str(label) or "decent-vla-fl")
    project = c.get("project") or env("WANDB_PROJECT")
    mode = c.get("mode") or env("WANDB_MODE") or "offline"
    enabled = c.get("enabled")
    if enabled is None:
        # Legacy inference: tracking is on iff a project is resolvable at all.
        enabled = bool(project)
    name = c.get("name") or env("WANDB_NAME") or rid
    return {
        "enabled": bool(enabled) and bool(project) and mode != "disabled",
        "project": project,
        "entity": c.get("entity") or env("WANDB_ENTITY"),
        "id": env("WANDB_RUN_ID") or re.sub(r"[^A-Za-z0-9_.-]", "-", str(name)),
        "name": name,
        "group": c.get("group") or env("WANDB_GROUP") or rid,
        "job_type": c.get("job_type") or env("WANDB_JOB_TYPE"),
        "mode": mode,
        "tags": c.get("tags") or None,
        "notes": c.get("notes") or None,
        "rid": rid,
    }


def wandb_round_log(label, round_num, client_losses, mean_loss=None, extra=None):
    """One wandb run per experiment; per-client loss as loss/<cid> + loss/mean."""
    global _WANDB_RUN, _WANDB_OFF
    if _WANDB_OFF:
        return
    s = _wandb_settings(label)
    if not s["enabled"]:
        return
    if _WANDB_RUN is None:
        try:
            import wandb
        except ImportError:
            logging.getLogger(__name__).warning("wandb requested but not installed; disabling")
            _WANDB_OFF = True
            return
        try:
            _WANDB_RUN = wandb.init(
                project=s["project"],
                entity=s["entity"],
                id=s["id"],
                name=s["name"],
                group=s["group"],
                job_type=s["job_type"],
                tags=s["tags"],
                notes=s["notes"],
                resume="allow",
                mode=s["mode"],
                dir=_log_dir(),
                config={"experiment": s["rid"]},
            )
        except Exception as exc:
            logging.getLogger(__name__).warning("wandb.init failed (%s); disabling wandb", exc)
            _WANDB_OFF = True
            return
    data = {"loss/" + str(cid): lo for cid, lo in (client_losses or {}).items()}
    if mean_loss is not None:
        data["loss/mean"] = mean_loss
    if extra:
        data.update(extra)
    if not data:
        return
    try:
        _WANDB_RUN.log(data, step=round_num)
    except Exception:
        pass

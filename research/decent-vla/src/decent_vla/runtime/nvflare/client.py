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

"""NVFlare Client-API loop: a thin envelope around the shared client session.

The loop runs inside ONE process for the whole job (in-process executor,
launch_once), so the warm ClientState in `runtime/client_session.py` —
backend, Adam moments, stepping scheduler, cycling dataloader — genuinely
persists across rounds: LocalRunner parity by construction. Config travels
via --config (ScriptRunner script_args) with the DECENT_VLA_CONFIG env var
as fallback; identity comes from the nvflare site name.
"""

from __future__ import annotations

import logging
import os

from decent_vla.runtime import client_session
from decent_vla.runtime.nvflare import codec

log = logging.getLogger(__name__)


def _configure(cfg) -> None:
    if cfg.fl is not None:
        codec.set_wire_dtype(cfg.fl.wire_dtype)


def restore_workdir(workdir: str | None, log=log) -> None:
    """Restore the job-submission cwd inside an nvflare-spawned process.

    Run configs use paths relative to the directory the run is launched from
    (backend.extra.train_config_path, relative fl.output_dir). NVFlare's
    simulator/POC processes run from a workspace cwd instead; restoring the
    submission cwd makes config-relative paths behave identically across
    runtimes. A missing dir (e.g. a prod site with a different layout) only
    warns — such deployments must use absolute paths in the config.
    """
    if not workdir:
        return
    if os.path.isdir(workdir):
        os.chdir(workdir)
    else:
        log.warning(
            "workdir %s does not exist on this site; config-relative paths "
            "will resolve against %s", workdir, os.getcwd()
        )


def cid_from_site(site_name: str, site_prefix: str) -> str:
    """Map an nvflare site name to a decent-vla cid.

    site_prefix "" means sites are named c0..cN-1 directly (what the simulator
    env does); a prefix like "site-" maps the conventional 1-based
    site-{i+1} to c{i}.
    """
    if not site_prefix:
        if not (site_name.startswith("c") and site_name[1:].isdigit()):
            raise ValueError(
                f"nvflare site name {site_name!r} is not a decent-vla cid; either name "
                f"sites c0..cN-1 or set fl.nvflare.site_prefix to their common prefix"
            )
        return site_name
    if not site_name.startswith(site_prefix) or not site_name[len(site_prefix):].isdigit():
        raise ValueError(
            f"nvflare site name {site_name!r} does not match fl.nvflare.site_prefix="
            f"{site_prefix!r} + <1-based index>"
        )
    return f"c{int(site_name[len(site_prefix):]) - 1}"


def resolve_config_path(arg: str | None) -> str:
    path = arg or os.environ.get("DECENT_VLA_CONFIG")
    if not path:
        raise SystemExit(
            "nvflare client: no run config — pass --config or set DECENT_VLA_CONFIG"
        )
    return path


def client_main(config_arg: str | None = None, workdir: str | None = None) -> None:
    """The multi-round Client-API loop (blocks until the server ends the job)."""
    import nvflare.client as flare

    restore_workdir(workdir, log=log)
    config_path = resolve_config_path(config_arg)
    flare.init()
    cfg, *_ = client_session.load_session(config_path, configure=_configure)
    cid = cid_from_site(flare.get_site_name(), cfg.fl.nvflare.site_prefix)

    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break
        global_state, plan = codec.flmodel_to_round(input_model)
        # Free the FLModel envelope as soon as its params are decoded: the
        # decoded tensors can share its numpy buffers, so holding both names
        # keeps one payload alive under two references.
        del input_model
        log.info(
            "nvflare client %s: round %d (%d local steps)",
            cid, plan.round_num, plan.local_steps,
        )
        update = client_session.handle_round(
            cid, config_path, global_state, plan, configure=_configure
        )
        # `global_state` is dead the moment the delta exists — release it before
        # encoding so the inbound broadcast and the outbound delta don't overlap.
        del global_state
        # Encode destructively, then send and drop. WITHOUT these deletes the
        # previous round's `update`, `global_state` and `input_model` stay bound
        # across the loop boundary and remain live through the NEXT round's
        # flare.receive() — measured at ~29 GB of dead payload per client at
        # pi0.5's 3.62B params, i.e. most of the reason this plane could not fit
        # in 125 GB of host RAM. `consume=True` additionally keeps the fp32 delta
        # from coexisting with its fp16 wire copy.
        out = codec.update_to_flmodel(update, consume=True)
        del update
        flare.send(out)
        del out, plan

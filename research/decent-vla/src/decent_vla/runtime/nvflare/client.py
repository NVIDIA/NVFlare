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
persists across rounds: LocalRunner parity by construction. The run config
travels inside the job (`job.build_job` packages it into the app's `config/`
folder) and is named by --config; identity comes from the nvflare site name.
"""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path

from decent_vla.runtime import client_session
from decent_vla.runtime.nvflare import codec

log = logging.getLogger(__name__)


def _configure(cfg) -> None:
    if cfg.fl is not None:
        codec.set_wire_dtype(cfg.fl.wire_dtype)


# The files a run config names that `job.staged_job` packages next to it: the
# lerobot training config and the held-out task spec. In the packaged config
# each is the file's bare name. The two functions below name the same fields,
# one in the raw YAML (submit side), one in the loaded config (site side).
def packaged_refs(raw: dict) -> list[tuple[str, dict, str]]:
    """Where a raw run config names its packaged files: (config key, the
    mapping that holds it, its field), for each place the loader accepts."""
    extra = (raw.get("backend") or {}).get("extra") or {}
    data = raw.get("data") or {}
    refs = [("backend.extra.train_config_path", extra, "train_config_path"), ("data.held_out", data, "held_out")]
    if isinstance(data.get("partitioning"), dict):  # the `data.partitioning: {scheme, held_out, ...}` form
        refs.append(("data.partitioning.held_out", data["partitioning"], "held_out"))
    return refs


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
    from decent_vla.configs import load_run_config

    cfg = load_run_config(str(config_path))
    resolve_packaged_paths(cfg, Path(config_path).parent)
    return cfg


def apply_site_paths(cfg, run_dir, workdir: str | None = None, output_dir: str | None = None,
                     log=log) -> None:
    """Point a run's site-local paths (data, outputs) at this site.

    A simulator job runs on the submitting host and carries that host's cwd
    (`workdir`) and, on the server, its absolute output folder: restore both, so
    relative paths behave as under local mode.

    A deployed job carries neither. Relative data paths (e.g. GR00T
    `dataset_paths`) resolve against DECENT_VLA_WORKDIR when the site sets it,
    otherwise against the process cwd; a relative `fl.output_dir` goes under
    this job's run folder, so each site keeps its outputs with the job.
    """
    if workdir:
        _chdir(workdir, "the submitting host's working directory")
        if output_dir:
            cfg.fl.output_dir = output_dir
        return
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


def resolve_config_path(arg: str | None, config_dir: str | None = None) -> str:
    path = arg or os.environ.get("DECENT_VLA_CONFIG")
    if not path:
        raise SystemExit(
            "nvflare client: no run config — pass --config or set DECENT_VLA_CONFIG"
        )
    if config_dir and not os.path.isabs(path):
        path = os.path.join(config_dir, path)
    return path


def client_main(config_arg: str | None = None, app_dir: str | None = None,
                workdir: str | None = None) -> None:
    """The multi-round Client-API loop (blocks until the server ends the job).

    `app_dir` is this site's job app folder: `custom/` holds the client
    script, `config/` the packaged run config, and the job's run folder is its
    parent (`<run_dir>/app_<site>`). `workdir` is set only for simulator runs.
    """
    import nvflare.client as flare

    config_path = resolve_config_path(config_arg, str(Path(app_dir) / "config") if app_dir else None)
    run_dir = Path(app_dir).parent if app_dir else Path.cwd()

    def configure(cfg) -> None:
        _configure(cfg)
        resolve_packaged_paths(cfg, Path(config_path).parent)
        apply_site_paths(cfg, run_dir=run_dir, workdir=workdir, log=log)

    flare.init()
    # A client that cannot start (missing data or weights, a bad site path)
    # still answers every round, with its error, rather than ending the
    # Client-API loop: the server then sees why it is missing.
    startup_error = None
    try:
        cfg, *_ = client_session.load_session(config_path, configure=configure)
        cid = cid_from_site(flare.get_site_name(), cfg.fl.nvflare.site_prefix)
    except Exception as exc:
        log.error("nvflare client %s: cannot start\n%s", flare.get_site_name(), traceback.format_exc())
        cid, startup_error = flare.get_site_name(), f"cannot start: {type(exc).__name__}: {exc}"

    # The reply to the last finished round, kept until the next round arrives.
    # The server sends a round again when a reply was late or lost (FLARE drops
    # a reply that arrives after its task timed out), or when it stopped
    # waiting at its quorum while this client was still training; either way
    # the client trained, and training again would step the warm optimizer,
    # scheduler and data iterator twice for one accepted update. So the same
    # round gets the stored reply back. Both cases need fl.round_timeout_s or
    # fl.min_clients; without either the server waits for every client and the
    # reply (one delta at the wire dtype) is not kept.
    keep_replies = startup_error is None and (
        cfg.fl.round_timeout_s is not None or cfg.fl.min_clients is not None
    )
    done: tuple[int, object] | None = None
    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break
        try:
            global_state, plan = codec.flmodel_to_round(input_model)
        except Exception as exc:
            log.error("nvflare client %s: cannot decode the round\n%s", cid, traceback.format_exc())
            del input_model
            flare.send(codec.error_flmodel(cid, -1, f"cannot decode the round: {type(exc).__name__}: {exc}"))
            continue
        # Free the FLModel envelope as soon as its params are decoded: the
        # decoded tensors can share its numpy buffers, so holding both names
        # keeps one payload alive under two references.
        del input_model
        if startup_error:
            del global_state
            flare.send(codec.error_flmodel(cid, plan.round_num, startup_error))
            continue
        if done is not None and plan.round_num <= done[0]:
            del global_state
            if plan.round_num < done[0]:
                why = f"round {plan.round_num} is older than this client's last round {done[0]}"
            elif done[1] is None:
                why = f"round {plan.round_num} again, and no stored reply to send"
            else:
                why = None
                log.warning(
                    "nvflare client %s: round %d again (its reply was late or lost); "
                    "sending the stored reply instead of training again", cid, plan.round_num,
                )
                flare.send(codec.copy_reply(done[1]))
            if why:
                flare.send(codec.error_flmodel(cid, plan.round_num, why))
            continue
        done = None  # the server has moved on; that reply can no longer be asked for
        log.info(
            "nvflare client %s: round %d (%d local steps)",
            cid, plan.round_num, plan.local_steps,
        )
        try:
            update = client_session.handle_round(
                cid, config_path, global_state, plan, configure=configure
            )
        except Exception as exc:
            # Report the failure instead of ending the loop: the client stays
            # available for the server's retry and later rounds, and the
            # server's round loop decides (min_clients, retry, RoundFailure).
            # handle_round has dropped the client's half-stepped state, so
            # the retry trains from a rebuilt one.
            log.error("nvflare client %s: round %d failed\n%s", cid, plan.round_num,
                      traceback.format_exc())
            del global_state
            flare.send(codec.error_flmodel(cid, plan.round_num, f"{type(exc).__name__}: {exc}"))
            continue
        # `global_state` is dead the moment the delta exists — release it before
        # encoding so the inbound broadcast and the outbound delta don't overlap.
        del global_state
        # Encode destructively, then send. WITHOUT these deletes the previous
        # round's `update`, `global_state` and `input_model` stay bound across
        # the loop boundary and remain live through the NEXT round's
        # flare.receive() — measured at ~29 GB of dead payload per client at
        # pi0.5's 3.62B params, i.e. most of the reason this plane could not fit
        # in 125 GB of host RAM. `consume=True` additionally keeps the fp32 delta
        # from coexisting with its fp16 wire copy. Only the wire copy is kept,
        # in `done`, and only when a resend can happen.
        out = codec.update_to_flmodel(update, consume=True, round_num=plan.round_num)
        del update
        # flare.send() drops out.params, so the kept reply is a copy (sharing the arrays).
        done = (plan.round_num, codec.copy_reply(out) if keep_replies else None)
        flare.send(out)
        del out, plan

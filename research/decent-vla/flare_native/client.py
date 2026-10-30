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

"""FLARE Client API entry point for local VLA training.

`job.py` ships this file in each app's `custom/` folder. FLARE executes it
in-process by default or as `python3 -u custom/client.py`; the warm policy,
optimizer, scheduler and data iterator therefore remain live across rounds.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


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
    if not site_name.startswith(site_prefix) or not site_name[len(site_prefix) :].isdigit():
        raise ValueError(
            f"nvflare site name {site_name!r} does not match fl.nvflare.site_prefix="
            f"{site_prefix!r} + <1-based index>"
        )
    return f"c{int(site_name[len(site_prefix):]) - 1}"


def resolve_config_path(arg: str | None, config_dir: str | None = None) -> str:
    path = arg or os.environ.get("DECENT_VLA_CONFIG")
    if not path:
        raise SystemExit("nvflare client: no run config — pass --config or set DECENT_VLA_CONFIG")
    if config_dir and not os.path.isabs(path):
        path = os.path.join(config_dir, path)
    return path


def client_main(config_arg=None, app_dir=None):
    """A warm policy trainer driven exclusively by the FLARE Client API."""
    import nvflare.client as flare
    import torch
    from nvflare.apis.fl_constant import FLMetaKey
    from nvflare.app_common.abstract.fl_model import FLModel, ParamsType

    from decent_vla._helpers import wire
    from decent_vla._helpers.builders import build_backend_factory, partition_for, resolve_device
    from decent_vla._helpers.client_checkpoint import resume_disabled, save_client_state
    from decent_vla._helpers.client_state import build_client_state, fast_forward_scheduler
    from decent_vla._helpers.site import apply_site_paths, load_job_config
    from decent_vla._helpers.training import seed_training, train_steps

    config_path = resolve_config_path(config_arg, str(Path(app_dir) / "config") if app_dir else None)
    cfg = load_job_config(config_path)
    apply_site_paths(cfg, Path(app_dir).parent if app_dir else Path.cwd())
    seed_training(cfg.seed, bool(cfg.backend.extra.get("deterministic", False)))
    flare.init()
    cid = cid_from_site(flare.get_site_name(), cfg.fl.nvflare.site_prefix)
    factory, cls = build_backend_factory(cfg, resolve_device(cfg.device))
    chunks, _ = partition_for(cfg, cls)
    state = None
    while flare.is_running():
        incoming = flare.receive()
        if incoming is None:
            break
        r = int(incoming.current_round)
        global_state = {k: torch.from_numpy(v) for k, v in incoming.params.items()}
        del incoming
        if state is None:
            seed_training(cfg.seed, bool(cfg.backend.extra.get("deterministic", False)))
            state = build_client_state(
                cid,
                factory,
                cfg.data,
                cfg.backend.optim,
                chunks[int(cid[1:])],
                start_round=r,
                local_steps=cfg.fl.local_steps,
                client_ckpt_dir=(
                    None
                    if resume_disabled() or not cfg.fl.persist_client_state
                    else Path(cfg.fl.output_dir) / "client_states"
                ),
            )
        expected_step = r * cfg.fl.local_steps
        if state.global_step < expected_step:
            fast_forward_scheduler(state.scheduler, expected_step - state.global_step)
            state.global_step = expected_step
        state.backend.set_state(global_state, scope="shared")
        loss, samples = train_steps(
            state.backend,
            state.dataloader,
            state.optimizer,
            state.scheduler,
            global_state,
            cfg.fl.local_steps,
            cfg.fl.strategy,
            cfg.backend.optim.grad_clip_norm,
        )
        state.global_step += cfg.fl.local_steps
        _, dt = wire.resolve(cfg.fl.wire_dtype)
        delta = {}
        for k, v in state.backend.get_state("shared").items():
            delta[k] = wire.cast(v.detach() - global_state[k].to(v.device), dt).cpu().numpy()
        del global_state
        if cfg.fl.persist_client_state:
            n = cfg.fl.save_every_n_rounds
            if (n and (r + 1) % n == 0) or r + 1 == cfg.fl.num_rounds:
                save_client_state(Path(cfg.fl.output_dir) / "client_states", state, r, cfg.fl.client_state_keep)
        # This standard FLARE weight carries DATASET SAMPLES, preserving the
        # authors' weighting; it is not necessarily the local optimizer-step count.
        flare.send(
            FLModel(
                params=delta,
                params_type=ParamsType.DIFF,
                current_round=r,
                metrics={"loss": loss},
                meta={FLMetaKey.NUM_STEPS_CURRENT_ROUND: samples},
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="decent-vla nvflare client")
    parser.add_argument(
        "--config",
        default=None,
        help="run config YAML, relative to the app's config/ folder (fallback: DECENT_VLA_CONFIG)",
    )
    args = parser.parse_args()

    # This file sits in <app>/custom/.
    client_main(args.config, app_dir=str(Path(__file__).resolve().parent.parent))


if __name__ == "__main__":
    main()

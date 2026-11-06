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
#
# Authors: Anbang Liu, Junhan Zhao, and Ziyue Xu

"""One external FLARE task runs the selected local model and training loss."""

from __future__ import annotations

import argparse
from pathlib import Path

from fabric_common import (
    SITES,
    check_environment,
    cpu_state,
    inside,
    install_feature_reader,
    load_core,
    load_setup,
    local_seed,
    make_spec,
    read_json,
    sha256_file,
    state_digest,
    training_args,
    write_json,
)
from fabric_progress import loader_progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    runtime = read_json(args.runtime)
    root = Path(runtime["root"]).resolve()
    fold_dir = inside(inside(root, "results"), runtime["fold_dir"])
    inside(fold_dir, args.runtime)
    setup = load_setup(root, runtime["experiment"], runtime.get("variant", "pooling"), runtime.get("top_k"))
    check_environment(root)
    core = load_core()

    import pandas as pd
    import torch

    import nvflare.client as flare
    from nvflare.app_common.abstract.fl_model import FLModel, ParamsType

    flare.init()
    try:
        site = flare.get_site_name()
        if site not in SITES:
            raise ValueError(f"Unexpected FLARE site: {site}")
        manifest = fold_dir / f"client_{site}_train_manifest.csv"
        if sha256_file(manifest) != runtime["manifest_sha256"][manifest.name]:
            raise ValueError("Client manifest changed after job preparation")
        client_df = pd.read_csv(manifest)
        manifest_root = Path(runtime["manifest_root"])
        train_args = training_args(root, setup, fold_dir.parent, manifest_root)
        if not torch.cuda.is_available():
            raise RuntimeError("Historical CUDA setup required; refusing silent CPU fallback")
        device = core.client_training.resolve_device(train_args)
        log_dir = fold_dir / "client_logs" / site
        log_dir.mkdir(parents=True, exist_ok=True)
        install_feature_reader(core, log_dir / "skipped_hdf5_rows.log")
        spec = make_spec(core, root, setup, fold_dir.parent, manifest_root)

        # launch_once=False gives each task its own process. No other client can
        # perturb this process's Python/NumPy/PyTorch random state.
        incoming = flare.receive()
        if incoming.params_type != ParamsType.FULL:
            raise ValueError("Only FULL model exchange is permitted")
        round_offset = runtime.get("resume_offset", 0)
        if type(round_offset) is not int or not 0 <= round_offset < setup["settings"]["rounds"]:
            raise ValueError("Invalid resume round offset")
        if incoming.total_rounds != setup["settings"]["rounds"] - round_offset:
            raise ValueError("Global round count changed")
        flare_round = incoming.current_round
        if type(flare_round) is not int or not 0 <= flare_round < incoming.total_rounds:
            raise ValueError(f"Invalid round: {flare_round}")
        round_idx = flare_round + round_offset
        global_state = cpu_state(incoming.params)
        input_digest = state_digest(global_state)
        expected = runtime.get("resume_state_sha256", runtime["initial_state_sha256"])
        if flare_round == 0 and input_digest != expected:
            raise ValueError("FLARE starting weights differ from the validated initialization/checkpoint")
        seed = local_seed(setup, runtime["fold"], round_idx, site)
        log_path = log_dir / f"round_{round_idx + 1}.json"
        if log_path.exists():
            raise FileExistsError(f"Refusing duplicate local training: {log_path}")
        label = (
            f"{setup['encoder']} fold_index={runtime['fold']} "
            f"round={round_idx + 1}/{train_args.rounds} site={site} local_train"
        )
        with loader_progress(
            core.client_training,
            epochs=train_args.local_epochs,
            label=label,
            log_path=log_dir / f"round_{round_idx + 1}_progress.log",
        ):
            state, log = core.client_training.train_local_client_final(
                core.experiment_utils.condition_for(spec),
                global_state,
                client_df,
                setup["model_name"],
                setup["input_dim"],
                train_args,
                device,
                seed=seed,
            )
        state = cpu_state(state)
        log.update(
            fold=runtime["fold"],
            round=round_idx + 1,
            site=site,
            model=setup["model_key"],
            algorithm="fedavg",
            seed=seed,
            local_epochs=train_args.local_epochs,
            input_state_sha256=input_digest,
            output_state_sha256=state_digest(state),
        )
        write_json(log_path, log)
        print(
            f"[FABRIC] fold={runtime['fold']} round={round_idx + 1} site={site} "
            f"patients={len(client_df)} loss={log['loss']:.6f}",
            flush=True,
        )
        flare.send(
            FLModel(
                params=state,
                params_type=ParamsType.FULL,
                current_round=flare_round,
                metrics={"train_loss": float(log["loss"])},
                meta={
                    "site": site,
                    "train_patients": len(client_df),
                    "seed": seed,
                    "local_epochs": train_args.local_epochs,
                    "input_state_sha256": input_digest,
                    "local_log": log,
                },
            )
        )
    finally:
        flare.shutdown()


if __name__ == "__main__":
    main()

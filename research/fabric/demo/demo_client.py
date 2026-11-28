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

"""Run one CPU training task on generated bags through the FLARE Client API."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from core.client_training import Condition, train_local_client_final
from demo_components import load_demo_runtime, read_demo_manifest, round_seed, training_args
from fabric_common import SITES, cpu_state, state_digest, write_json

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    runtime = load_demo_runtime(args.runtime)
    directory = Path(runtime["demo_dir"])
    torch.set_num_threads(1)
    flare.init()
    try:
        site = flare.get_site_name()
        if site not in SITES:
            raise ValueError("Unexpected demonstration site")
        incoming = flare.receive()
        number = incoming.current_round
        if (
            incoming.params_type != ParamsType.FULL
            or incoming.total_rounds != 2
            or type(number) is not int
            or not 0 <= number < 2
        ):
            raise ValueError("Unexpected demonstration task")
        frame = read_demo_manifest(runtime, site)
        if len(frame) != runtime["client_counts"][site] or set(frame["site"]) != {site}:
            raise ValueError("Synthetic client assignment changed")
        state = cpu_state(incoming.params)
        input_digest = state_digest(state)
        if number == 0 and input_digest != runtime["initial_state_sha256"]:
            raise ValueError("Unexpected demonstration initialization")
        seed = round_seed(runtime, number, site)
        condition = Condition("demo", "Synthetic FABRIC demonstration", "fedavg", "demo", 0.0, Path())
        print(f"[FABRIC demo] {site} round={number + 1}/2: training {len(frame)} synthetic cases", flush=True)
        state, log = train_local_client_final(
            condition,
            state,
            frame,
            runtime["model_name"],
            runtime["input_dim"],
            training_args(runtime),
            torch.device("cpu"),
            seed,
        )
        state = cpu_state(state)
        write_json(
            directory / "client_logs" / site / f"round_{number + 1}.json",
            {
                **log,
                "round": number + 1,
                "site": site,
                "seed": seed,
                "input_state_sha256": input_digest,
                "output_state_sha256": state_digest(state),
            },
        )
        flare.send(
            FLModel(
                params=state,
                params_type=ParamsType.FULL,
                current_round=number,
                metrics={"train_loss": float(log["loss"])},
                meta={"site": site, "train_patients": len(frame), "seed": seed, "input_state_sha256": input_digest},
            )
        )
    finally:
        flare.shutdown()


if __name__ == "__main__":
    main()

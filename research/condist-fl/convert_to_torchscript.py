# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from src.utils.get_model import get_model


def run_convert(args):
    if not Path(args.config).exists():
        raise ValueError(f"Config file {args.config} does not exists.")

    if not Path(args.weights).exists():
        raise ValueError(f"Checkpoint file {args.weights} does not exists.")

    app = "app_" + args.app
    if app == "app_server":
        with open(args.config) as f:
            config = json.load(f)
            config = [c for c in config["components"] if c["id"] == "model"][0]
            config["name"] = config["path"].split(".")[-1]
            config["path"] = ".".join(config["path"].split(".")[:-1])
    else:
        with open(args.config) as f:
            config = json.load(f)
            config = config["model"]

    ckpt = torch.load(args.weights)
    state_dict = ckpt["model"]

    model = get_model(config)
    model.load_state_dict(state_dict)
    model = model.cuda().eval()

    sample_data = torch.rand([1, 1, 224, 224, 64]).cuda()
    traced_module = torch.jit.trace(model, sample_data)

    torch.jit.save(traced_module, args.output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", type=str, help="Path to the config file.")
    parser.add_argument("--weights", "-w", type=str, help="Path to the saved model checkpoint.")
    parser.add_argument(
        "--app",
        "-a",
        type=str,
        choices=["server", "kidney", "liver", "pancreas", "spleen"],
        help="Select app to convert checkpoint.",
    )
    parser.add_argument("--output", "-o", type=str, help="Output result JSON.")
    args = parser.parse_args()

    run_convert(args)

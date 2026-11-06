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

import os
import runpy
from pathlib import Path

import torch


def main():
    import nvflare.client as flare

    receive_model, send_model = flare.receive, flare.send
    folder = Path("fedce_evidence")
    folder.mkdir(exist_ok=True)
    current_round = None
    scores = {}

    def receive(*args, **kwargs):
        nonlocal current_round
        model = receive_model(*args, **kwargs)
        if model is None:
            return None
        current_round = model.current_round
        scores[current_round] = []
        torch.save({"params": model.params, "meta": model.meta}, folder / f"{current_round}_input.pt")
        return model

    def send(model, *args, **kwargs):
        torch.save(
            {
                "params": model.params,
                "metrics": model.metrics,
                "meta": model.meta,
                "pid": os.getpid(),
                "validations": scores[current_round],
            },
            folder / f"{current_round}_output.pt",
        )
        return send_model(model, *args, **kwargs)

    flare.receive, flare.send = receive, send
    namespace = runpy.run_path(str(Path(__file__).with_name("shipped_client.py")), run_name="fedce_shipped")
    validate_model = namespace["_validate"]

    def validate(*args, **kwargs):
        score = validate_model(*args, **kwargs)
        scores[current_round].append(float(score))
        return score

    namespace["main"].__globals__["_validate"] = validate
    namespace["main"](namespace["_parse_args"]())


if __name__ == "__main__":
    main()

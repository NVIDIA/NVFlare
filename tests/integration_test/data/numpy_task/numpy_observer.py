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

import numpy as np


def main():
    import nvflare.client as flare

    receive_model, send_model = flare.receive, flare.send
    folder = Path("numpy_evidence")
    folder.mkdir(exist_ok=True)
    current_round = None

    def receive(*args, **kwargs):
        nonlocal current_round
        model = receive_model(*args, **kwargs)
        current_round = model.current_round
        np.savez(folder / f"{current_round}_input.npz", **model.params)
        return model

    def send(model, *args, **kwargs):
        np.savez(
            folder / f"{current_round}_output.npz",
            **model.params,
            pid=os.getpid(),
            weight_mean=model.metrics["weight_mean"],
            params_type=str(model.params_type),
        )
        return send_model(model, *args, **kwargs)

    flare.receive, flare.send = receive, send
    runpy.run_path(str(Path(__file__).with_name("shipped_client.py")), run_name="__main__")


if __name__ == "__main__":
    main()

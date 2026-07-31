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

"""Produce a streamed result from an external-process Swarm trainer."""

import numpy as np

import nvflare.client as flare


def main():
    flare.init()

    while flare.is_running():
        input_model = flare.receive()
        weights = np.asarray(input_model.params["numpy_key"])

        # A 4 MiB result makes the DownloadService path visible while keeping the
        # reproducer CPU-only and independent of any dataset or ML framework.
        trained_weights = np.full((1024, 1024), weights.mean() + 1, dtype=np.float32)
        flare.send(
            flare.FLModel(
                params={"numpy_key": trained_weights},
                params_type="FULL",
                metrics={"accuracy": 1.0},
                meta={"NUM_STEPS_CURRENT_ROUND": 1},
            )
        )
        print(
            f"swarm external trainer sent round {input_model.current_round} " f"result shape={trained_weights.shape}",
            flush=True,
        )


if __name__ == "__main__":
    main()

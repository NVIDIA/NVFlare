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

"""Externally owned NumPy trainer for the Attach + CCWF acceptance test."""

import argparse

import numpy as np

import nvflare.client as flare
from nvflare.app_common.np.constants import NPConstants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    flare.init(config_file=args.config)
    site_name = flare.get_site_name()
    delta = float(site_name.rsplit("-", maxsplit=1)[-1])
    last_params = None

    try:
        while flare.is_running():
            model = flare.receive()
            if model is None:
                break

            if flare.is_train():
                incoming = np.asarray(model.params[NPConstants.NUMPY_KEY])
                update = np.full_like(incoming, delta)
                last_params = incoming + update
                flare.send(
                    flare.FLModel(
                        params={NPConstants.NUMPY_KEY: update},
                        params_type=flare.ParamsType.DIFF,
                        metrics={"val_accuracy": float(last_params.mean())},
                        current_round=model.current_round,
                    )
                )
            elif flare.is_submit_model():
                if last_params is None:
                    raise RuntimeError("submit_model received before the trainer completed a train task")
                flare.send(
                    flare.FLModel(
                        params={NPConstants.NUMPY_KEY: last_params},
                        params_type=flare.ParamsType.FULL,
                    )
                )
            elif flare.is_evaluate():
                weights = np.asarray(model.params[NPConstants.NUMPY_KEY])
                flare.send(flare.FLModel(metrics={"val_mean": float(weights.mean())}))
            else:
                raise RuntimeError(f"unexpected task {flare.get_task_name()!r}")
    finally:
        flare.shutdown()


if __name__ == "__main__":
    main()

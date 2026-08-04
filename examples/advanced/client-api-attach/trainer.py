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

"""An externally owned trainer; it may start before or after the NVFlare job."""

import argparse

import numpy as np

import nvflare.client as flare
from nvflare.app_common.np.constants import NPConstants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="attach_profile_shared_file.json")
    args = parser.parse_args()

    flare.init(config_file=args.config)
    print(f"Attached as {flare.get_site_name()} to job {flare.get_job_id()}")
    while flare.is_running():
        model = flare.receive()
        weights = model.params[NPConstants.NUMPY_KEY]
        updated = np.asarray(weights) + 1
        flare.send(
            flare.FLModel(
                params={NPConstants.NUMPY_KEY: updated},
                metrics={"weight_mean": float(updated.mean())},
                current_round=model.current_round,
            )
        )


if __name__ == "__main__":
    main()

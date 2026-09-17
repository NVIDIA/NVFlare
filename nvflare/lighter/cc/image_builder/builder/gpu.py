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

"""Guest-only CUDA readiness; GPU appraisal belongs to Trustee's composite transaction."""

from .common import require, run


def readiness(config, ready):
    if config.get("gpu") != "nvidia_cc":
        return
    if ready:
        devices = run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"], timeout=10).splitlines()
        require(len(devices) == config["gpu_count"], "GPU count changed after composite appraisal")
    run(["nvidia-smi", "conf-compute", "-srs", "1" if ready else "0"], timeout=10)

# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import subprocess
from typing import List


def get_host_gpu_ids() -> List:
    try:
        process = subprocess.Popen(
            ["nvidia-smi", "--list-gpus"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        result = process.communicate()
        rc = process.returncode
        if rc > 0:
            raise Exception("Failed to get host gpu device Ids", result[0])
        else:
            # 'GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-xxxx-xxxx-xxxx-xxx)\n'
            if result[0].startswith("GPU"):
                gpus = result[0].split("\n")
                gpu_ids = [int(gpu.split(":")[0].split(" ")[1]) for gpu in gpus[:-1]]
            else:
                gpu_ids = []
    except FileNotFoundError as e:
        print(f"Failed to get gpu device Ids {e}")
        print("Assume no gpu device in the host")
        gpu_ids = []

    return gpu_ids

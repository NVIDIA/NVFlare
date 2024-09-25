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

import numpy as np
import torch


def convert_global_to_ckpt(global_model_filepath: str, ckpt_path: str):
    """Helper function to convert global models saved by NVFlare to NeMo ckpt format"""

    nvflare_ckpt = torch.load(global_model_filepath)
    if "train_conf" in nvflare_ckpt:
        print("Loaded NVFlare global checkpoint with train_conf", nvflare_ckpt["train_conf"])

    assert (
        "model" in nvflare_ckpt
    ), f"Expected global model to contain a 'model' key but it only had {list(nvflare_ckpt.keys())}"
    global_weights = nvflare_ckpt["model"]

    torch.save({"state_dict": global_weights}, ckpt_path)

    print(f"Saved NeMo ckpt with {len(global_weights)} entries to {ckpt_path}")

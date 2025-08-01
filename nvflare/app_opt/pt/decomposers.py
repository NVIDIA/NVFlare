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

from typing import Any, Optional

import torch
from safetensors.torch import _remove_duplicate_names, load_file, save_file

import nvflare.fuel.utils.fobs.dots as dots
from nvflare.fuel.utils.fobs.decomposers.via_file import ViaFileDecomposer


class SerializationModule(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.register_buffer("saved_tensor", tensor)


def _safe_save(state_dict, filename: str) -> Optional[dict]:
    """Save model weights with the safetensors format.
    The model weights may contain tensors with shared memory. In this case, save_file won't work.
    We first try to find and remove such tensors, and then save the remaining tensors with save_file.
    We then return the information about the removed tensors as a dict.
    The key of the dict is the name of the tensor kept in the weights.
    The value is a list of tensor names that are to be substituted by the kept tensor.

    For example, the state_dict contains multiple tensors:

    {
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "t4": t4
    }

    Suppose tensors t1, t2 and t3 are shared, the state_dict after removing shared tensors will look like this:

    {
        "t1": t1,
        "t4": t4
    }

    And the removed tensors dict looks like this:
    {
        "t1": ["t2", "t3"]
    }

    Args:
        state_dict: the model weights to be saved
        filename: name of the file

    Returns: a dict that contains removed tensor info

    """
    to_removes = _remove_duplicate_names(state_dict)
    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            del state_dict[to_remove]
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    save_file(state_dict, filename)
    if to_removes:
        # to_removes is dict-like but not a simple dict
        return {k: v for k, v in to_removes.items()}
    else:
        return None


class TensorDecomposer(ViaFileDecomposer):

    def supported_type(self):
        return torch.Tensor

    def dump_to_file(self, items: dict, path: str, fobs_ctx: dict):
        try:
            meta = _safe_save(items, path)
            self.logger.info(f"dumping {len(items)} tensors to file {path}: removed tensor info {meta}")
            return path, meta
        except Exception as e:
            self.logger.error(f"exception dumping tensors to file: {e}")
            raise e

    def load_from_file(self, path: str, fobs_ctx: dict, meta: dict = None) -> Any:
        items = load_file(path)
        self.logger.debug(f"got {len(items)} tensors from file {path}")
        if meta:
            # the meta keeps names of removed tensors and the name of the tensor for them
            for kept, removed_group in meta.items():
                for r in removed_group:
                    items[r] = items[kept]
        return items

    def get_bytes_dot(self) -> int:
        return dots.TENSOR_BYTES

    def get_file_dot(self) -> int:
        return dots.TENSOR_FILE

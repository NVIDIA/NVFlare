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

from typing import Any

import torch

import nvflare.fuel.utils.fobs.dots as dots
from nvflare.fuel.utils.fobs.decomposers.via_file import ViaFileDecomposer


class SerializationModule(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.register_buffer("saved_tensor", tensor)


class TensorDecomposer(ViaFileDecomposer):
    """We first planned to use safetensors' "save_file" and "load_file" to save and load tensors.
    Unfortunately that does not work if there are shared memory among different tensors.
    Safetensors suggests to use "save_model" and "load_model" instead, which does not work for us either.
    This is because both methods require a model (nn.Module) object that defines model architecture.
    Though this could be done on the sending side, there is no way for the receiving side to have the model
    object during the recomposition process.

    We decided to use torch's "save" and "load" methods, which work even if tensors have shared memory among them.
    NOTE: the "save" method does NOT involve pickle when saving only model weights, which is what we do.
    """

    def supported_type(self):
        return torch.Tensor

    def supported_dots(self):
        return [dots.TENSOR_BYTES, dots.TENSOR_FILE]

    def dump_to_file(self, items: dict, path: str):
        self.logger.debug(f"dumping {len(items)} tensors to file {path}")
        try:
            torch.save(items, path)
        except Exception as e:
            self.logger.error(f"exception dumping tensors to file: {e}")
            raise e

    def load_from_file(self, path: str) -> Any:
        items = torch.load(path, weights_only=True)
        self.logger.debug(f"got {len(items)} tensors from file {path}")
        return items

    def get_bytes_dot(self) -> int:
        return dots.TENSOR_BYTES

    def get_file_dot(self) -> int:
        return dots.TENSOR_FILE

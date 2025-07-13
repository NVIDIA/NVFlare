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
from safetensors.torch import load_file, save_file

import nvflare.fuel.utils.fobs.dats as dats
from nvflare.fuel.utils.fobs.decomposers.via_file import ViaFileDecomposer


class SerializationModule(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.register_buffer("saved_tensor", tensor)


class TensorDecomposer(ViaFileDecomposer):
    def supported_type(self):
        return torch.Tensor

    def supported_dats(self):
        return [dats.LOCAL_TENSOR, dats.REMOTE_TENSOR]

    def dump_to_file(self, items: dict, path: str):
        print(f"SafeTensor: dumping {len(items)} tensors to file {path}")
        save_file(items, path)

    def load_from_file(self, path: str) -> Any:
        items = load_file(path)
        print(f"SafeTensor: got {len(items)} tensors from file {path}")
        return items

    def get_local_dat(self) -> int:
        return dats.LOCAL_TENSOR

    def get_remote_dat(self) -> int:
        return dats.REMOTE_TENSOR

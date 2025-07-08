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

from nvflare.app_common.decomposers.via_file import ViaFileDecomposer


class SerializationModule(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.register_buffer("saved_tensor", tensor)


class TensorDecomposer(ViaFileDecomposer):
    def supported_type(self):
        return torch.Tensor

    def dump_to_file(self, target: Any, path: str):
        tensors = {"tensor": target}
        save_file(tensors, path)

    def load_from_file(self, path: str) -> Any:
        loaded_tensors = load_file(path)
        return loaded_tensors.get("tensor")

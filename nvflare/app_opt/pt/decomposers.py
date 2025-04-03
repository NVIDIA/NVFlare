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

from io import BytesIO
from typing import Any

import numpy as np
import torch

from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.datum import DatumManager
from nvflare.fuel.utils.fobs.decomposer import Externalizer, Internalizer


class SerializationModule(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.register_buffer("saved_tensor", tensor)


class TensorDecomposer(fobs.Decomposer):
    def supported_type(self):
        return torch.Tensor

    def decompose(self, target: torch.Tensor, manager: DatumManager = None) -> Any:
        externalizer = Externalizer(manager)
        if target.dtype == torch.bfloat16:
            return self._jit_serialize(target, externalizer)
        else:
            return self._numpy_serialize(target, externalizer)

    def recompose(self, data: Any, manager: DatumManager = None) -> torch.Tensor:
        internalizer = Internalizer(manager)
        buf = internalizer.internalize(data["buffer"])
        if data["dtype"] == "torch.bfloat16":
            return self._jit_deserialize(buf)
        return self._numpy_deserialize(buf)

    @staticmethod
    def _numpy_serialize(tensor: torch.Tensor, externalizer) -> dict:
        stream = BytesIO()
        # supported ScalarType, use numpy to avoid Pickle
        array = tensor.detach().cpu().numpy()
        np.save(stream, array, allow_pickle=False)
        return {
            "buffer": externalizer.externalize(stream.getvalue()),
            "dtype": str(tensor.dtype),
        }

    @staticmethod
    def _numpy_deserialize(data: Any) -> torch.Tensor:
        stream = BytesIO(data)
        array = np.load(stream, allow_pickle=False)
        return torch.from_numpy(array)

    @staticmethod
    def _jit_serialize(tensor: torch.Tensor, externalizer) -> dict:
        stream = BytesIO()
        # unsupported ScalarType by numpy, use torch.jit to avoid Pickle
        module = SerializationModule(tensor)
        torch.jit.save(torch.jit.script(module), stream)
        return {
            "buffer": externalizer.externalize(stream.getvalue()),
            "dtype": str(tensor.dtype),
        }

    @staticmethod
    def _jit_deserialize(data: Any) -> torch.Tensor:
        stream = BytesIO(data)
        loaded_module = torch.jit.load(stream)
        return loaded_module.saved_tensor

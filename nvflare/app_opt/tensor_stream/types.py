# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Union

import torch

TENSORS_CHANNEL = "tensor_stream"

TensorsMap = Union[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]


class TensorCustomKeys:
    TASK_ID = "task_id"
    SAFE_TENSORS_PROP_KEY = "_safe_tensors_blob_"


class TensorTopics:
    TASK_RESULT = "task_result"
    TASK_DATA = "task_data"


class TensorBlobKeys:
    SAFETENSORS_BLOB = "safetensors_blob"
    TENSOR_KEYS = "tensor_keys"
    ROOT_KEY = "root_key"
    TASK_ID = "task_id"


class TensorEventTypes:
    SEND_TENSORS_FOR_TASK_DATA = "SEND_TENSORS_FOR_TASK_DATA"

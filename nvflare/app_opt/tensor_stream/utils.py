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

import numpy as np
import torch

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import SERVER_SITE_NAME
from nvflare.apis.shareable import Shareable

from .types import TensorTopics


def clean_task_data(fl_ctx: FLContext):
    """Clean the task data in the FLContext.

    Args:
        fl_ctx (FLContext): The FLContext to clean the task data from.
    """
    task_data: Shareable = fl_ctx.get_prop(FLContextKey.TASK_DATA)
    # set data to empty to avoid sending large data within the task data
    task_data["DXO"]["data"] = {}
    fl_ctx.set_prop(FLContextKey.TASK_DATA, value=task_data, private=True, sticky=False)


def clean_task_result(fl_ctx: FLContext):
    """Clean the task result in the FLContext.

    Args:
        fl_ctx (FLContext): The FLContext to clean the task result from.
    """
    task_result: Shareable = fl_ctx.get_prop(FLContextKey.TASK_RESULT)
    # set data to empty to avoid sending large data within the task result
    task_result["DXO"]["data"] = {}
    fl_ctx.set_prop(FLContextKey.TASK_RESULT, value=task_result, private=True, sticky=False)


def get_topic_for_ctx_prop_key(ctx_prop_key: str) -> str:
    """Get the topic based on the context property key.

    Args:
        ctx_prop_key (str): The context property key.

    Returns:
        str: The topic associated with the context property key.
    """
    if ctx_prop_key == FLContextKey.TASK_DATA:
        return TensorTopics.TASK_DATA
    elif ctx_prop_key == FLContextKey.TASK_RESULT:
        return TensorTopics.TASK_RESULT
    else:
        raise ValueError(f"Unsupported context property key: {ctx_prop_key}")


def get_targets_for_ctx_and_prop_key(fl_ctx: FLContext, ctx_prop_key: str) -> list[str]:
    """Get the peer identity name from the FLContext.

    Args:
        fl_ctx (FLContext): The FLContext for the current operation.
    Returns:
        list[str]: The identity name(s) of the peer(s).
    """
    if ctx_prop_key == FLContextKey.TASK_DATA:
        return [fl_ctx.get_peer_context().get_identity_name()]
    elif ctx_prop_key == FLContextKey.TASK_RESULT:
        return [SERVER_SITE_NAME]
    else:
        raise ValueError(f"Unsupported context property key: {ctx_prop_key}")


def to_numpy_recursive(obj: Union[torch.Tensor, dict[str, torch.Tensor]]) -> Union[dict[str, np.ndarray], np.ndarray]:
    """Recursively convert objects with a `numpy` method to numpy arrays."""
    if hasattr(obj, "numpy"):
        return obj.numpy()
    elif isinstance(obj, dict):
        return {k: to_numpy_recursive(v) for k, v in obj.items()}
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")


def to_torch_recursive(
    obj: Union[np.ndarray, dict[str, np.ndarray]], device: torch.device = "cpu"
) -> Union[dict[str, torch.Tensor], torch.Tensor]:
    """Recursively convert numpy arrays to torch tensors."""
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).to(device)
    elif isinstance(obj, dict):
        return {k: to_torch_recursive(v, device) for k, v in obj.items()}
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")


def validate_torch_dict_params_recursive(tensor_dict: dict):
    """Recursively validate that all values in the dictionary are torch tensors."""
    if not isinstance(tensor_dict, dict):
        raise ValueError(f"Expected a dictionary, but got {type(tensor_dict)}")

    for key, value in tensor_dict.items():
        if isinstance(value, dict):
            validate_torch_dict_params_recursive(value)
        elif not isinstance(value, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor for key '{key}', but got {type(value)}")


def validate_numpy_dict_params_recursive(tensor_dict: dict):
    """Recursively validate that all values in the dictionary are numpy arrays."""
    if not isinstance(tensor_dict, dict):
        raise ValueError(f"Expected a dictionary, but got {type(tensor_dict)}")

    for key, value in tensor_dict.items():
        if isinstance(value, dict):
            validate_numpy_dict_params_recursive(value)
        elif not isinstance(value, np.ndarray):
            raise ValueError(f"Expected np.ndarray for key '{key}', but got {type(value)}")

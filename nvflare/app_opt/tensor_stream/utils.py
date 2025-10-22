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

from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch

from nvflare.apis.dxo import DXO, DataKind, from_shareable
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
    # keep only the non-tensor in the task data since tensors are sent separately
    new_task_data = copy_non_tensor_params(task_data["DXO"]["data"])
    task_data["DXO"]["data"] = new_task_data
    fl_ctx.set_prop(FLContextKey.TASK_DATA, value=task_data, private=True, sticky=False)


def clean_task_result(fl_ctx: FLContext):
    """Clean the task result in the FLContext.

    Args:
        fl_ctx (FLContext): The FLContext to clean the task result from.
    """
    task_result: Shareable = fl_ctx.get_prop(FLContextKey.TASK_RESULT)
    # keep only the non-tensor in the task result since tensors are sent separately
    new_task_result = copy_non_tensor_params(task_result["DXO"]["data"])
    task_result["DXO"]["data"] = new_task_result
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
    """Recursively convert torch tensors to numpy arrays with minimal memory duplication.

    Note: For CPU tensors, .numpy() returns a view sharing memory with the original tensor (zero-copy).
    For GPU tensors, data must be moved to CPU first, which creates a copy.
    Only the dictionary structure is duplicated, not the underlying tensor data (for CPU tensors).

    Args:
        obj: A torch.Tensor or dict containing torch.Tensors (possibly nested)

    Returns:
        A numpy array or dict containing numpy arrays. Tensor data is shared where possible (CPU tensors).
    """
    if hasattr(obj, "numpy"):
        # .numpy() returns a view for CPU tensors (no data copy)
        # For GPU tensors, must call .cpu() first which creates a copy
        if obj.is_cuda:
            return obj.cpu().numpy()
        return obj.numpy()
    elif isinstance(obj, dict):
        # Create new dict structure but reuse converted tensors (which share memory with originals)
        return {k: to_numpy_recursive(v) for k, v in obj.items()}
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")


def get_dxo_from_ctx(fl_ctx: FLContext, ctx_prop_key: FLContextKey, tasks: list[str]) -> DXO:
    """Extract model parameters from the FLContext based on the provided property key.

    Args:
        fl_ctx (FLContext): The FLContext containing the data.

    Returns:
        dict[str, torch.Tensor]: A dictionary of data extracted from the FLContext.
    """
    task_name = fl_ctx.get_prop(FLContextKey.TASK_NAME)
    if not task_name:
        raise ValueError("No task name found in FLContext.")

    if task_name not in tasks:
        raise ValueError(f"Task name '{task_name}' not part of configured tasks: {tasks}")

    task: Shareable = fl_ctx.get_prop(ctx_prop_key)
    if task is None:
        raise ValueError(f"No task found in FLContext. Looked for for shareable in '{ctx_prop_key}'.")

    dxo = from_shareable(task)
    if dxo.data_kind not in (DataKind.WEIGHTS, DataKind.WEIGHT_DIFF):
        raise ValueError(f"Skipping task, data kind is not WEIGHTS or WEIGHT_DIFF: {dxo.data_kind}")

    return dxo


def chunk_tensors_from_params(
    params: Dict[str, Union[torch.Tensor, dict]],
    parent_keys: Optional[List[str]] = None,
    chunk_size: Optional[int] = 10,
) -> Iterator[Tuple[Tuple[str], Dict[str, torch.Tensor]]]:
    """
    Generator that yields tensors grouped by their immediate parent dictionary keys.

    Args:
        params: Dictionary with string keys and values that are either torch.Tensor or nested dicts.
        parent_keys: List of keys representing the current path (internal use, defaults to empty).
        chunk_size: Optional maximum number of tensors to yield at once per parent.

    Yields:
        A tuple containing:
        - List of parent keys (excluding the tensor key itself).
        - Dictionary mapping tensor key names to torch.Tensor instances.
    """
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer or None")

    if parent_keys is None:
        parent_keys = []

    tensors = {}
    for key, value in params.items():
        if isinstance(value, torch.Tensor):
            tensors[key] = value
        elif isinstance(value, np.ndarray):
            tensors[key] = torch.from_numpy(value)
        elif isinstance(value, dict):
            yield from chunk_tensors_from_params(value, parent_keys + [key], chunk_size)

    if tensors:
        if chunk_size is None or chunk_size >= len(tensors):
            yield parent_keys, tensors
        else:
            keys = list(tensors.keys())
            for i in range(0, len(keys), chunk_size):
                chunk_keys = keys[i : i + chunk_size]
                chunk_tensors = {k: tensors[k] for k in chunk_keys}
                yield tuple(parent_keys), chunk_tensors


def update_params_with_tensors(
    params: Dict, parents: List[str], tensors: Dict[str, torch.Tensor], to_ndarray: bool = False
) -> None:
    """
    Updates the nested dictionary `params` at the location specified by
    `parents` with the provided tensor values from `tensors`.

    If `to_ndarray` is True, tensors are converted to numpy ndarrays before insertion.

    Args:
        params: The dictionary to update (possibly nested).
        parents: List of keys that specify the nested path within `params`.
        tensors: Dictionary mapping keys to torch.Tensor instances.
        to_ndarray: Whether to convert tensors to numpy arrays before updating.
    """
    cur = params
    for key in parents:
        if key not in cur:
            cur[key] = {}
        elif not isinstance(cur[key], dict):
            raise ValueError(f"Expected dict at key '{key}', but found {type(cur[key])}")
        cur = cur[key]

    for k, tensor in tensors.items():
        if to_ndarray:
            cur[k] = tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
        else:
            cur[k] = tensor


def merge_params_dicts(
    base_params: Dict[str, dict],
    new_params: Dict[str, dict],
    to_ndarray: bool = False,
) -> Dict[str, dict]:
    """
    Merges two nested dictionaries of parameters.

    Args:
        base_params: The base dictionary to merge into.
        new_params: The new dictionary whose values will overwrite those in base_params.

    Returns:
        The merged dictionary with values from new_params overwriting those in base_params.
    """
    for key, value in new_params.items():
        if key in base_params and isinstance(base_params[key], dict) and isinstance(value, dict):
            merge_params_dicts(base_params[key], value)
        else:
            if to_ndarray and isinstance(value, torch.Tensor):
                base_params[key] = value.cpu().numpy() if value.is_cuda else value.numpy()
            else:
                base_params[key] = value
    return base_params


def copy_non_tensor_params(params: Dict[str, dict]) -> Dict[str, dict]:
    """Recursively copy non-tensor parameters in the given dictionary.

    Args:
        params: The dictionary of parameters to copy from.

    Returns:
        A new dictionary containing only non-tensor parameters.
    """
    non_tensor_params = {}
    for key, value in params.items():
        if isinstance(value, dict):
            nested_non_tensors = copy_non_tensor_params(value)
            if nested_non_tensors:
                non_tensor_params[key] = nested_non_tensors
        elif not isinstance(value, (torch.Tensor, np.ndarray)):
            non_tensor_params[key] = value
    return non_tensor_params

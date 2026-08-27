# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import Optional, Union

from nvflare.apis.analytix import AnalyticsDataType as AnalyticsDataType
from nvflare.app_common.abstract.fl_model import FLModel as FLModel
from nvflare.app_common.abstract.fl_model import ParamsType as ParamsType
from nvflare.app_opt.hf import patch as patch
from nvflare.app_opt.hf.api import hf_is_running as is_running
from nvflare.client.api import get_config as get_config
from nvflare.client.api import get_job_id as get_job_id
from nvflare.client.api import get_site_name as get_site_name
from nvflare.client.api import get_task_name as get_task_name
from nvflare.client.api import init as _client_api_init
from nvflare.client.api import is_evaluate as is_evaluate
from nvflare.client.api import is_submit_model as is_submit_model
from nvflare.client.api import is_train as is_train
from nvflare.client.api import log as log
from nvflare.client.api import receive as receive
from nvflare.client.api import send as send
from nvflare.client.api import shutdown as shutdown
from nvflare.client.api import system_info as system_info
from nvflare.client.decorator import evaluate as evaluate
from nvflare.client.decorator import train as train
from nvflare.client.ipc.ipc_agent import IPCAgent as IPCAgent

_MULTIRANK_SIZE_ENV_VARS = ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS")


def _get_initialized_distributed_rank():
    try:
        import torch.distributed as dist
    except ImportError:
        return None

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return int(dist.get_rank())
    return None


def _environment_declares_multirank() -> bool:
    for name in _MULTIRANK_SIZE_ENV_VARS:
        try:
            if int(os.environ.get(name, "1") or 1) > 1:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _resolve_init_rank(rank):
    if rank is not None:
        return rank

    distributed_rank = _get_initialized_distributed_rank()
    if distributed_rank is not None:
        return distributed_rank

    if _environment_declares_multirank():
        global_rank = os.environ.get("RANK")
        try:
            normalized_rank = int(global_rank)
        except (TypeError, ValueError):
            normalized_rank = -1
        if normalized_rank < 0:
            raise RuntimeError(
                "Hugging Face Client API detected a multi-process launch but global RANK is unavailable or invalid; "
                "initialize torch.distributed or export a valid non-negative global RANK before flare.init()"
            )
        return str(normalized_rank)
    return None


def init(rank: Optional[Union[str, int]] = None, config_file: Optional[str] = None):
    """Initialize the Client API with Hugging Face distributed-rank validation."""
    return _client_api_init(rank=_resolve_init_rank(rank), config_file=config_file)


__all__ = [
    "AnalyticsDataType",
    "FLModel",
    "IPCAgent",
    "ParamsType",
    "evaluate",
    "get_config",
    "get_job_id",
    "get_site_name",
    "get_task_name",
    "init",
    "is_evaluate",
    "is_running",
    "is_submit_model",
    "is_train",
    "log",
    "patch",
    "receive",
    "send",
    "shutdown",
    "system_info",
    "train",
]

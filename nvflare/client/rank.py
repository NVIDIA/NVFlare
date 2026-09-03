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
import sys
from typing import Optional, Union

MULTIRANK_SIZE_ENV_VARS = ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE")
CLIENT_API_PROCESS_COUNT_ENV_VAR = "NVFLARE_CLIENT_API_PROCESS_COUNT"
SLURM_TASK_COUNT_ENV_VAR = "SLURM_NTASKS"
SLURM_PROCESS_ID_ENV_VAR = "SLURM_PROCID"


def _environment_declares_single_client_api_process() -> bool:
    process_count = os.environ.get(CLIENT_API_PROCESS_COUNT_ENV_VAR)
    if process_count is None:
        return False
    try:
        return int(process_count) == 1
    except (TypeError, ValueError):
        return False


def environment_declares_multirank() -> bool:
    """Return whether a supported launcher declares more than one process."""
    client_api_process_count = os.environ.get(CLIENT_API_PROCESS_COUNT_ENV_VAR)
    if client_api_process_count is not None:
        return not _environment_declares_single_client_api_process()
    for name in MULTIRANK_SIZE_ENV_VARS:
        try:
            if int(os.environ.get(name, "1") or 1) > 1:
                return True
        except (TypeError, ValueError):
            continue
    # SLURM_NTASKS alone can be inherited by a single-process child. Require the
    # per-process marker before treating the allocation size as a trainer launch.
    if SLURM_PROCESS_ID_ENV_VAR in os.environ:
        try:
            return int(os.environ.get(SLURM_TASK_COUNT_ENV_VAR, "1") or 1) > 1
        except (TypeError, ValueError):
            return False
    return False


def get_initialized_torch_distributed_rank() -> Optional[int]:
    """Return an initialized Torch process-group rank without importing Torch."""
    dist = sys.modules.get("torch.distributed")
    if dist is None:
        return None
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return int(dist.get_rank())
    return None


def normalize_process_rank(rank: Union[str, int]) -> str:
    """Return a canonical non-negative decimal process rank."""
    if isinstance(rank, bool) or not isinstance(rank, (str, int)):
        raise ValueError(f"rank must be a string or an integer but got {type(rank)}")
    try:
        normalized_rank = int(rank)
    except ValueError as e:
        raise ValueError(f"rank must be a non-negative integer but got {rank!r}") from e
    if normalized_rank < 0:
        raise ValueError(f"rank must be a non-negative integer but got {rank!r}")
    return str(normalized_rank)


def resolve_process_rank(rank: Optional[Union[str, int]] = None) -> str:
    """Resolve an explicit, initialized Torch, Client API process-count, or launcher-provided global rank."""
    if rank is not None:
        return normalize_process_rank(rank)

    distributed_rank = get_initialized_torch_distributed_rank()
    if distributed_rank is not None:
        return normalize_process_rank(distributed_rank)

    # A launcher can designate one Client API participant even when its process
    # inherits an unrelated global rank from the surrounding study environment.
    if _environment_declares_single_client_api_process():
        return "0"

    environment_rank = os.environ.get("RANK")
    if environment_rank is not None:
        return normalize_process_rank(environment_rank)

    if environment_declares_multirank():
        raise RuntimeError(
            "NVFlare Client API detected a multi-process launch but global RANK is unavailable; "
            "initialize torch.distributed or export a valid global RANK before flare.init()"
        )
    return "0"

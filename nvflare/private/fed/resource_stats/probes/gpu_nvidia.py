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


"""Count the NVIDIA GPU inventory selected by FLARE's integer visibility rules."""

import os

from nvflare.fuel.utils.gpu_utils import get_host_gpu_ids, get_visible_gpu_ids


def probe_gpu():
    """Return a verified count, or unavailable for unsupported selectors/query failures."""
    visible = get_visible_gpu_ids()
    if visible == []:
        return {"count": 0}
    if visible is None and "CUDA_VISIBLE_DEVICES" in os.environ:
        return None
    try:
        host_ids = get_host_gpu_ids(timeout=2.0)
    except Exception:
        return None
    if not host_ids:
        return None
    visible = get_visible_gpu_ids(host_ids)
    return {"count": len(visible)} if visible is not None else None

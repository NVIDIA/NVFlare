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

"""FL runtime: client loop, local-mode runner, partitioning, NVFlare runtime."""
from __future__ import annotations

from decent_vla.runtime.client_loop import run_client_round
from decent_vla.runtime.local_runner import LocalRunner
from decent_vla.runtime.partitioning import iid_partition, task_partition

__all__ = ["run_client_round", "LocalRunner", "iid_partition", "task_partition"]

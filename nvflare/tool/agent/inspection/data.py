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

"""Adapter over the existing one-walk dataset inspector."""

from pathlib import Path

from nvflare.tool.agent.dataset_inspect import _inspect_dataset_with_audit
from nvflare.tool.agent.inspection.result import data_result


def inspect_data_target(target: Path, *, max_files: int, max_file_bytes: int) -> dict:
    dataset, audit = _inspect_dataset_with_audit(target, max_files, max_file_bytes)
    return data_result(target, dataset, audit, max_files=max_files, max_file_bytes=max_file_bytes)

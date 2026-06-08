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

"""Auto-FL utilities for agent-assisted NVFlare job optimization."""

__all__ = [
    "AUTOFL_CONFIG_SCHEMA_VERSION",
    "DeterministicJobImporter",
    "JobImportError",
    "dump_autofl_yaml",
    "import_job_to_autofl_config",
]


def __getattr__(name):
    if name in __all__:
        from nvflare.app_common.autofl import job_importer

        return getattr(job_importer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

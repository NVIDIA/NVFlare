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

"""Backend-independent, project-relative configuration paths."""

import os

from nvflare.lighter.constants import PropKey


def to_abs_path(yaml_path, file_path):
    if not isinstance(yaml_path, str) or not yaml_path:
        raise ValueError("Invalid input: 'yaml_path' must be a non-empty string.")
    if not isinstance(file_path, str) or not file_path:
        raise ValueError("Invalid input: 'file_path' must be a non-empty string.")
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(yaml_path)), file_path))


def resolve_cc_config(project, value):
    if not isinstance(value, str) or not value:
        raise ValueError("cc_config must be a non-empty YAML path")
    origin = project.get_prop(PropKey.PROJECT_FILE)
    if not origin and not os.path.isabs(value):
        raise ValueError("Relative cc_config requires prepare_project(..., project_file=...) or an absolute path")
    return to_abs_path(str(origin or value), value)

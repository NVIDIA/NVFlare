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

from nvflare.apis.fl_constant import AdminCommandNames
from nvflare.security.security import COMMAND_CATEGORIES, CommandCategory


def test_resource_query_commands_use_view_authorization_category():
    assert COMMAND_CATEGORIES[AdminCommandNames.GET_JOB_RESOURCES] == CommandCategory.VIEW
    assert COMMAND_CATEGORIES[AdminCommandNames.GET_STUDY_RESOURCES] == CommandCategory.VIEW

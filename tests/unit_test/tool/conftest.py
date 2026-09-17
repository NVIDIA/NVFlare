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

import pytest

from nvflare.tool import cli_output


@pytest.fixture(autouse=True)
def reset_cli_output_state():
    cli_output.set_output_format("txt")
    cli_output.set_connect_timeout(5.0)
    yield
    cli_output.set_output_format("txt")
    cli_output.set_connect_timeout(5.0)

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

"""Bound malformed-input parsing in a subprocess so regressions cannot hang pytest."""

import subprocess
import sys


def test_repeated_unterminated_comment_starts_do_not_stall_config_parsing():
    code = (
        "from nvflare.fuel.utils.json_config_loader import JsonConfigLoader\n"
        "try:\n"
        "    JsonConfigLoader().load_config_from_str('/*x' * 350000)\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('unterminated comment was accepted')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr

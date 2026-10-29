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

"""CoCo external-runtime contracts; run serially outside unit workers."""

import sys

from tests.coco_test_utils.process import run_coco_command


def test_tdx_target_import_does_not_require_tomllib():
    script = "import sys; sys.modules['tomllib'] = None; from nvflare.lighter.cc_provision import workload_launch_profile; from nvflare.lighter.cc_provision.kata_runtime_profile import runtime_target; assert runtime_target('kata-qemu-tdx')['cpu_tee'] == 'tdx'"
    result = run_coco_command([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

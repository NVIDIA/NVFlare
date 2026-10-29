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

import subprocess

import pytest

from nvflare.fuel.utils import gpu_utils


@pytest.mark.parametrize("returncode", [1, -9, -15])
def test_nvidia_smi_nonzero_exit(monkeypatch, returncode):
    monkeypatch.setattr(gpu_utils, "has_nvidia_smi", lambda: True)
    monkeypatch.setattr(
        gpu_utils.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], returncode, stdout="", stderr="failed"),
    )

    with pytest.raises(Exception, match="Failed to call nvidia-smi"):
        gpu_utils.get_host_gpu_ids()

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

from types import SimpleNamespace

import pytest

from nvflare.fuel.utils import gpu_utils

GPU = "GPU-aaaaaaaa-0000-0000-0000-000000000001"


def test_discovery_uses_host_uuids_and_current_mig_modes_with_bounded_query(monkeypatch):
    calls = []
    monkeypatch.setattr(gpu_utils, "has_nvidia_smi", lambda: True)

    def query(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=GPU.upper() + ", Disabled\n", stderr="")

    monkeypatch.setattr(gpu_utils.subprocess, "run", query)
    assert gpu_utils.get_host_gpu_uuid_modes() == {GPU: "Disabled"}
    assert calls == [
        (
            ["nvidia-smi", "--query-gpu=uuid,mig.mode.current", "--format=csv,noheader"],
            {"capture_output": True, "text": True, "timeout": 10},
        )
    ]


def test_discovery_without_nvidia_smi_fails_instead_of_claiming_empty_inventory(monkeypatch):
    monkeypatch.setattr(gpu_utils, "has_nvidia_smi", lambda: False)
    with pytest.raises(RuntimeError, match="nvidia-smi is required"):
        gpu_utils.get_host_gpu_uuid_modes()


@pytest.mark.parametrize("row", ["0, Disabled", "MIG-invalid, Enabled", "GPU-invalid, Disabled", GPU])
def test_discovery_rejects_malformed_device_identity_or_row(monkeypatch, row):
    monkeypatch.setattr(gpu_utils, "use_nvidia_smi", lambda *args, **kwargs: [row])
    with pytest.raises(ValueError):
        gpu_utils.get_host_gpu_uuid_modes()

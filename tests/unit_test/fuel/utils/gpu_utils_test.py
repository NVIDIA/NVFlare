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

from nvflare.fuel.utils.gpu_utils import get_cuda_device_environment


def test_cuda_visibility_preserves_assignment_and_ordinal_order():
    assert get_cuda_device_environment(("GPU-a", "MIG-b", 2)) == {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "GPU-a,MIG-b,2",
    }
    assert get_cuda_device_environment(())["CUDA_VISIBLE_DEVICES"] == ""

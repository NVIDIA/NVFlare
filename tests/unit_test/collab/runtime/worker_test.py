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
import sys

import pytest


def test_worker_registers_tensor_decomposer_for_raw_pytorch():
    pytest.importorskip("torch")
    code = """
import torch

from nvflare.collab.runtime.worker.worker import _register_tensor_decomposer
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.datum import DatumManager

fobs.reset()
_register_tensor_decomposer()
value = torch.tensor([1.0, 2.0])
manager = DatumManager()
restored = fobs.deserialize(fobs.serialize(value, manager), manager)
assert torch.equal(restored, value)
"""
    subprocess.run([sys.executable, "-c", code], check=True)

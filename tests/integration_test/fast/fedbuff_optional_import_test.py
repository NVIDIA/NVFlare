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

"""Exercise a fresh interpreter import without optional tensor frameworks."""

import subprocess
import sys


def test_async_imports_without_tensor_framework_or_synthesis():
    code = """
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('torch', 'torchvision', 'kornia'):
            raise ImportError('optional dependency blocked')
sys.meta_path.insert(0, Block())
from nvflare.collab import FedBuffRecipe
from nvflare.collab.fedbuff_recipe import FedBuffResult, FedBuffTrainingCoordinator
from examples.advanced.collab.numpy_async.job import NumpyModelOps
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120, capture_output=True, text=True)

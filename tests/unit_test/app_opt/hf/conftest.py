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

import sys

import pytest


def _is_managed_module(module_name: str) -> bool:
    return (
        module_name == "transformers"
        or module_name.startswith("transformers.")
        or module_name == "peft"
        or module_name.startswith("peft.")
        or module_name == "nvflare.app_opt.hf"
        or module_name.startswith("nvflare.app_opt.hf.")
        or module_name == "nvflare.client.hf"
        or module_name.startswith("nvflare.client.hf.")
    )


@pytest.fixture(autouse=True)
def restore_hf_optional_module_cache():
    """Keep fake HF optional-dependency modules from leaking between tests."""

    snapshot = {name: module for name, module in sys.modules.items() if _is_managed_module(name)}
    yield
    for name in list(sys.modules):
        if _is_managed_module(name):
            sys.modules.pop(name, None)
    sys.modules.update(snapshot)

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
from contextlib import contextmanager
from pathlib import Path

FEDCORE_DIR = Path(__file__).resolve().parents[3] / "research" / "fedcore"


def _is_fedcore_module(name: str) -> bool:
    return name in {"evaluate", "job", "model", "run_demo", "src"} or name.startswith("src.")


@contextmanager
def fedcore_import_context():
    """Import standalone FedCoRe modules without leaking generic module names into other tests."""

    original_modules = {name: module for name, module in sys.modules.items() if _is_fedcore_module(name)}
    for name in original_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(FEDCORE_DIR))
    try:
        yield FEDCORE_DIR
    finally:
        sys.path.remove(str(FEDCORE_DIR))
        for name in list(sys.modules):
            if _is_fedcore_module(name):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)

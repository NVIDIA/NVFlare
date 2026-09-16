#!/usr/bin/env python3
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

"""Refuse unsafe host dump/swap settings before starting the key broker."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from builder.common import BuildError, require, validate_core_policy

try:
    validate_core_policy()
    require(len(Path("/proc/swaps").read_text().splitlines()) == 1, "Disable Trustee host swap")
except (BuildError, OSError):
    raise SystemExit("Trustee requires swap disabled and no piped core collector") from None

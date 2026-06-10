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

import json


def canonical_json(value) -> str:
    """Serialize a JSON-compatible value to its canonical string form.

    Canonical form is deterministic so it can be hashed or signed:
    keys are sorted, separators carry no whitespace, and non-ASCII characters
    are kept as-is (UTF-8) rather than escaped.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

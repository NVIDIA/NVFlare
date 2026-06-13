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

"""Select the single staged SDK wheel for a benchmark Docker stage."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: select_sdk_wheel.py <variant> <output-path>", file=sys.stderr)
        return 2
    variant = sys.argv[1]
    output_path = Path(sys.argv[2])
    package = os.environ.get("SDK_PACKAGE_NAME", "SDK")
    wheels = sorted(Path("/tmp/wheels").glob("*.whl"))
    if len(wheels) != 1:
        print(
            f"Expected exactly one staged {package} {variant} wheel in /tmp/wheels, "
            f"found {len(wheels)}: {[path.name for path in wheels]}",
            file=sys.stderr,
        )
        return 2
    output_path.write_text(str(wheels[0]), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

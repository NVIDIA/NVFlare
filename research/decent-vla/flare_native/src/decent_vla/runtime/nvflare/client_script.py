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

"""Entry script NVFlare's ScriptRunner ships to each client site.

`job.build_job` packages this FILE into the job app's `custom/` folder (the
run config goes in the app's `config/` folder), and NVFlare executes it inside
the client: in-process by default, or as `python3 -u custom/client_script.py`
in a subprocess. So it must stay a standalone script: argparse + a main guard,
everything else delegated to the installed decent_vla package.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="decent-vla nvflare client")
    parser.add_argument(
        "--config",
        default=None,
        help="run config YAML, relative to the app's config/ folder (fallback: DECENT_VLA_CONFIG)",
    )
    args = parser.parse_args()

    from decent_vla.runtime.nvflare.client import client_main

    # This file sits in <app>/custom/.
    client_main(args.config, app_dir=str(Path(__file__).resolve().parent.parent))


if __name__ == "__main__":
    main()

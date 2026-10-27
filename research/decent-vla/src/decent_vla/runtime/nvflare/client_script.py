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

ScriptRunner copies this FILE into the job folder and executes it inside the
client (in-process executor by default), so it must stay a standalone script:
argparse + a main guard, everything else delegated to the installed
decent_vla package.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="decent-vla nvflare client")
    parser.add_argument("--config", default=None, help="run config YAML (fallback: DECENT_VLA_CONFIG)")
    parser.add_argument("--workdir", default=None, help="job-submission cwd to restore")
    args = parser.parse_args()

    from decent_vla.runtime.nvflare.client import client_main

    client_main(args.config, workdir=args.workdir)


if __name__ == "__main__":
    main()

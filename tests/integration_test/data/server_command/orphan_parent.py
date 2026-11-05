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

"""Leave a detached child after a clean parent exit for the cleanup regression."""

import subprocess
import sys

child = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import os,sys,time; from pathlib import Path; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); print('ready', flush=True); time.sleep(60)",
        sys.argv[1],
    ],
    start_new_session=True,
    stdout=subprocess.PIPE,
    text=True,
)
assert child.stdout.readline().strip() == "ready"
child.stdout.close()

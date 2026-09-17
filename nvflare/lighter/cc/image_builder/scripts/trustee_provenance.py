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

"""Record the exact reviewed source patch and built KBS binary for deployment."""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from builder.common import digest_file, require, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("binary", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
commit = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
diff = subprocess.check_output(["git", "-C", str(args.source), "diff", "HEAD", "--binary"])
require(
    diff and diff == (args.source / "cvm-boundary.patch").read_bytes(),
    "Source differs from the recorded boundary patch",
)
guest = args.source / "cvm_guest"
guest_commit = subprocess.check_output(["git", "-C", str(guest), "rev-parse", "HEAD"], text=True).strip()
require(guest_commit == "591d0bb45cd7a2c66f3778428940c40f7eec3b7d", "Wrong guest-components revision")
guest_diff = subprocess.check_output(["git", "-C", str(guest), "diff", "HEAD", "--binary"])
require(guest_diff == (args.source / "cvm_guest.patch").read_bytes(), "Guest sources differ from the recorded patch")
write_json(
    args.output,
    {
        "trustee_commit": commit,
        "trustee_patch_digest": hashlib.sha256(diff).hexdigest(),
        "binary_sha256": digest_file(args.binary),
    },
)

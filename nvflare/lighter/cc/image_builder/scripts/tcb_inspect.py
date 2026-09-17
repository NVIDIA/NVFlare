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

"""Inspect candidate TDX TCB fields; this command never approves or imports them."""

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from builder.builder import verify_reference
from builder.common import read_json, require

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("evidence", type=Path)
args = parser.parse_args()
evidence = read_json(args.evidence)
require(evidence["platform"] == "intel_tdx", "This inspector expects TDX reference evidence")
verify_reference("intel_tdx", evidence)
report = base64.b64decode(evidence["report"], validate=True)
print(
    json.dumps(
        {
            "unapproved_candidate_tcb": {
                "mr_seam": [report[280:328].hex()],
                "tcb_svn": [report[264:280].hex()],
                "xfam": [report[520:528].hex()],
            },
            "review_required": "Validate platform endorsements and TCB status; obtain advisory IDs from a real signed-quote appraisal. This output is not an approved reference file.",
        },
        indent=2,
    )
)

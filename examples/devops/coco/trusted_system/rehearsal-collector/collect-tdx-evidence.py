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

"""Collect a fresh TDX quote and CCEL through the guest-local AA REST API.

This program does NOT approve the evidence. The trusted host must independently
verify the quote, nonce, InitData, Intel collateral, and complete CCEL replay.
"""

import base64
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

MAX_EVIDENCE_BYTES = 32 * 1024 * 1024
EVIDENCE_URL = "http://127.0.0.1:8006/aa/evidence"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("guest evidence endpoint must not redirect")


def collect(challenge_path=Path("/challenge/request-data.bin")):
    challenge = challenge_path.read_bytes()
    # REST passes the decoded UTF8 query string directly as report_data. It
    # neither base64-decodes nor hashes it. 32 random bytes rendered as 64 hex
    # characters preserve 256 bits of nonce entropy and fit TDX REPORT_DATA.
    if not re.fullmatch(rb"[0-9a-f]{64}", challenge):
        raise ValueError("challenge must contain 64 lowercase hexadecimal ASCII bytes")
    url = EVIDENCE_URL + "?" + urllib.parse.urlencode({"runtime_data": challenge.decode("ascii")})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(url, timeout=180) as response:
        payload = response.read(MAX_EVIDENCE_BYTES + 1)
    if not payload or len(payload) > MAX_EVIDENCE_BYTES:
        raise ValueError("empty or oversized evidence response")
    evidence = json.loads(payload)
    if not isinstance(evidence, dict) or set(evidence) != {"quote", "cc_eventlog"}:
        raise ValueError("AA did not return raw TDX quote and cc_eventlog")
    for name in ("quote", "cc_eventlog"):
        value = evidence[name]
        if not isinstance(value, str) or not value or not base64.b64decode(value, validate=True):
            raise ValueError(f"AA returned missing or malformed {name}")
    return payload


def main():
    evidence = collect()
    print("COCO_TDX_EVIDENCE_V1=" + base64.b64encode(evidence).decode("ascii"), flush=True)
    # Allow the trusted host to capture the live QEMU launch arguments before
    # normal Pod teardown removes the sandbox process. No shell or devices.
    time.sleep(45)


if __name__ == "__main__":
    main()

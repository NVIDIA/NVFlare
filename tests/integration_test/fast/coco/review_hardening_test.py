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

"""CoCo external-runtime contracts; run serially outside unit workers."""

import base64
import hashlib
import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nvflare.lighter.cc_provision import kbs_audience
from tests.coco_test_utils.process import run_coco_command


def test_audience_probe_signs_tokens_and_checks_all_four_http_results(tmp_path):
    key = tmp_path / "admin.key"
    run_coco_command(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(key)], check=True, capture_output=True)
    observed = []

    def respond(request, timeout):
        header = request.get_header("Authorization")
        claims = json.loads(base64.urlsafe_b64decode(header.split(".")[1] + "===")) if header else {}
        observed.append(claims)
        if claims.get("aud") != ["KBS"]:
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)
        response = MagicMock()
        response.__enter__.return_value.status = 200
        return response

    opener = MagicMock()
    opener.open.side_effect = respond
    with (
        patch.object(kbs_audience.ssl, "create_default_context"),
        patch.object(kbs_audience.urllib.request, "build_opener", return_value=opener),
    ):
        kbs_audience.probe("https://secure.example", "ca.pem", key)
    assert [item.get("aud") for item in observed] == [["KBS"], ["not-KBS"], None, None]
    assert observed[2]["iss"] == "TrusteeInDocker"


@pytest.mark.parametrize("change", [None, "profile", "pod", "duplicate", "traversal", "oversized", "symlink"])
def test_handoff_generation_checks_local_policy_outputs(tmp_path, change):
    script_path = Path(__file__).resolve().parents[4] / "examples/devops/coco/admin/40-create-handoffs.sh"
    marker = "python3 - \"${OUTPUT_DIR}\" <<'PY'\n"
    check = script_path.read_text().split(marker, 1)[1].split("\nPY", 1)[0]
    names = {
        "approved-workload-launch-profile.json",
        "pod.yaml",
        "final-initdata.toml",
        "generated-policy.rego",
        "image-security-policy.json",
        "release-authorization.json",
        "resource-policy-fragment.rego",
        "kata-policy-security-fix.txt",
    }
    for name in names:
        (tmp_path / name).write_bytes(("generated " + name).encode())
    manifest = "".join(
        hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() + "  " + name + "\n" for name in sorted(names)
    )
    (tmp_path / "policy-SHA256SUMS").write_text(manifest)
    if change in {"profile", "pod"}:
        name = "approved-workload-launch-profile.json" if change == "profile" else "pod.yaml"
        (tmp_path / name).write_text("changed after generation")
    elif change == "duplicate":
        (tmp_path / "policy-SHA256SUMS").write_text(manifest + manifest.splitlines()[0] + "\n")
    elif change == "traversal":
        (tmp_path / "policy-SHA256SUMS").write_text("0" * 64 + "  ../outside\n")
    elif change == "oversized":
        (tmp_path / "policy-SHA256SUMS").write_text("x" * 8193)
    elif change == "symlink":
        (tmp_path / "pod.yaml").unlink()
        (tmp_path / "pod.yaml").symlink_to(tmp_path / "final-initdata.toml")
    result = run_coco_command([sys.executable, "-", str(tmp_path)], input=check, capture_output=True, text=True)
    assert (result.returncode == 0) is (change is None), result.stderr

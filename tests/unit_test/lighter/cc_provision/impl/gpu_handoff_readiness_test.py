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

"""Ready-Trustee CLI accepts opaque credentials without local deployment inspection."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nvflare.lighter.cc_provision import workload_release
from tests.unit_test.lighter.cc_provision.impl.existing_trustee_integration_test import (
    INITIAL_POLICY,
    make_handoff,
    update_manifest,
)

COCO = Path(__file__).resolve().parents[5] / "examples/devops/coco"


def run_installer(tmp_path, runtime, pin_mode="omitted", invalid_target=False):
    kit = tmp_path / "kit"
    (kit / "lib").mkdir(parents=True)
    for name in ("12-install-trusted-service-handoff.sh", "lib/common.sh"):
        shutil.copyfile(COCO / "service" / name, kit / name)
    shutil.copyfile(COCO / "shared/lib/common-base.sh", kit / "lib/common-base.sh")
    (kit / "lib/install-workload-release.py").symlink_to(COCO / "service/lib/install-workload-release.py")
    incoming, pin = make_handoff(tmp_path / "handoff", runtime)
    if invalid_target:
        path = incoming / "release-authorization.json"
        auth = json.loads(path.read_text())
        auth["gpu"] = "unapproved"
        path.write_text(json.dumps(auth))
        pin = update_manifest(incoming)
    current = tmp_path / "operator-current.rego"
    current.write_bytes(INITIAL_POLICY)
    certificate = tmp_path / "ca.pem"
    certificate.write_text("existing trusted CA fixture")
    credential = tmp_path / "credential"
    credential.write_text("opaque-server-issued-credential-with-custom-role")
    credential.chmod(0o644)
    client = tmp_path / "kbs-client"
    client.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "assert args[:2] == ['--url', 'https://localhost:9544']\n"
        "assert Path(args[6]).read_text() == 'opaque-server-issued-credential-with-custom-role'\n"
        "operation = args[7]\n"
        "record = {'operation': operation, 'arguments': args[8:]}\n"
        "with Path(os.environ['EVENTS']).open('a') as stream: stream.write(json.dumps(record) + '\\n')\n"
        "print('vendor client output may include secret resource bytes')\n"
    )
    client.chmod(0o755)
    values = {
        "KBS_URL": "https://localhost:9544",
        "TRUSTEE_PUBLIC_CERT": str(certificate),
        "ADMIN_TOKEN": str(credential),
        "KBS_CLIENT": str(client),
        "STATE_DIR": str(tmp_path / "state"),
        "CURRENT_RESOURCE_POLICY_FILE": str(current),
    }
    (kit / "platform.env").write_text("".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items()))
    arguments = ["bash", str(kit / "12-install-trusted-service-handoff.sh"), str(incoming)]
    if pin_mode != "omitted":
        arguments.append({"empty": "", "valid": pin, "malformed": "bad", "mismatch": "0" * 64}[pin_mode])
    events = tmp_path / "events"
    result = subprocess.run(
        arguments, capture_output=True, text=True, timeout=20, env={**os.environ, "EVENTS": str(events)}
    )
    records = [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
    return result, records, current


@pytest.mark.parametrize("runtime", workload_release.RUNTIME_TARGETS)
@pytest.mark.parametrize("pin_mode", ["omitted", "empty", "valid"])
def test_ready_trustee_only_needs_api_access_current_export_and_handoff(tmp_path, runtime, pin_mode):
    result, records, current = run_installer(tmp_path, runtime, pin_mode)
    assert result.returncode == 0, result.stderr
    assert [record["operation"] for record in records] == [
        "set-resource",
        "set-resource",
        "set-resource",
        "set-resource-policy",
    ]
    assert current.read_bytes().startswith(INITIAL_POLICY.rstrip())
    assert "secret resource bytes" not in result.stdout + result.stderr
    assert "opaque-server-issued" not in result.stdout + result.stderr
    assert "Trustee accepted resource uploads" in result.stdout


@pytest.mark.parametrize("pin_mode", ["malformed", "mismatch"])
def test_cli_rejects_invalid_supplied_pin_before_any_api_request(tmp_path, pin_mode):
    result, records, current = run_installer(tmp_path, "kata-qemu-snp", pin_mode)
    assert result.returncode != 0
    assert not records
    assert "manifest pin" in result.stderr
    assert current.read_bytes() == INITIAL_POLICY


@pytest.mark.parametrize("runtime", workload_release.RUNTIME_TARGETS)
def test_cli_rejects_invalid_target_before_any_api_request(tmp_path, runtime):
    result, records, current = run_installer(tmp_path, runtime, invalid_target=True)
    assert result.returncode != 0
    assert not records
    assert current.read_bytes() == INITIAL_POLICY

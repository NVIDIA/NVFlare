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

"""Exercise real embedded generators/validators without deploying any services."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = (ROOT / "service/12-install-trusted-service-handoff.sh").read_text()
TEMPLATE = ROOT / "service/policies/workload-resource-policy.rego.template"


def blocks(script):
    return re.findall(r"<<'PY'\n(.*?)\nPY\n", script, re.S)


def python_block(block, *args):
    return subprocess.run([sys.executable, "-", *map(str, args)], input=block, text=True, capture_output=True)


class WorkloadHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.fragment = self.work / "resource-policy-fragment.rego"
        self.rendered = self.work / "rendered.rego"
        self.registry = "secure.unit.local:5000"
        self.repository = self.registry + "/workloads/demo"
        self.image = self.repository + "@sha256:" + "a" * 64
        self.paths = ["default/" + kind + "/review-test" for kind in ("image-key", "sig-public-key", "security-policy")]
        self.generator = next(
            b for b in blocks((ROOT / "admin/30-generate-pod-and-policies.sh").read_text()) if "fragment = f" in b
        )
        self.generate(["/app/run"])
        (self.work / "cosign.pub").write_text("-----BEGIN PUBLIC KEY-----\nfixture-only\n")
        (self.work / "image-security-policy.json").write_text(
            json.dumps(
                {
                    "default": [{"type": "reject"}],
                    "transports": {
                        "docker": {
                            self.repository: [
                                {
                                    "type": "sigstoreSigned",
                                    "keyPath": "kbs:///" + self.paths[1],
                                    "signedIdentity": {"type": "matchRepository"},
                                }
                            ]
                        }
                    },
                }
            )
        )

    def generate(self, args):
        result = python_block(
            self.generator, self.work, "review-test", self.image, json.dumps(args), "b" * 64, *self.paths
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def validate(self):
        return python_block(blocks(INSTALLER)[0], self.work, self.registry, TEMPLATE, self.rendered)

    def test_generated_handoff_matches_trusted_template(self):
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.fragment.read_bytes(), self.rendered.read_bytes())
        self.assertIn('"${ACTIVE_POLICY}" "${REVIEWED_FRAGMENT}"', INSTALLER)

    def test_reject_arbitrary_and_targeted_rules(self):
        original = self.fragment.read_text()
        for rule in (
            "allow := true",
            'allow := true if { data["resource-path"] == ["default", "image-key", "other-release"] }',
            'allow := true if { input.submods.cpu0["ear.veraison.annotated-evidence"]["init_data"] == "c" }',
        ):
            with self.subTest(rule=rule):
                self.fragment.write_text(original + "\n" + rule + "\n")
                result = self.validate()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("differs from the secure-services template", result.stderr)
                self.assertFalse(self.rendered.exists())

    def test_reject_removed_identity_check(self):
        self.fragment.write_text(
            self.fragment.read_text().replace(
                '    cpu["ear.veraison.annotated-evidence"]["init_data"] == wo_review_test_expected_initdata\n', ""
            )
        )
        self.assertNotEqual(self.validate().returncode, 0)
        self.assertFalse(self.rendered.exists())

    def test_reject_cross_release_path_in_authorization(self):
        path = self.work / "release-authorization.json"
        auth = json.loads(path.read_text())
        auth["kbs_resource_paths"][0] = "default/image-key/other-release"
        path.write_text(json.dumps(auth))
        self.assertNotEqual(self.validate().returncode, 0)
        self.assertFalse(self.rendered.exists())

    def test_arguments_are_data_not_rego_or_template_variables(self):
        self.generate(["/app/run", '${prefix}\n"}\nallow := true\n#'])
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.fragment.read_bytes(), self.rendered.read_bytes())
        self.assertNotIn("\nallow := true\n", self.rendered.read_text())

    @unittest.skipUnless(shutil.which("opa"), "OPA CLI unavailable; run with stage-01 OPA on PATH for semantic tests")
    def test_opa_behavior_and_cross_release_denial(self):
        self.assertEqual(self.validate().returncode, 0)
        candidate = self.work / "candidate.rego"
        candidate.write_text("package policy\nimport rego.v1\ndefault allow := false\n" + self.rendered.read_text())
        subprocess.run(["opa", "check", "--strict", str(candidate)], check=True, capture_output=True)
        semantic = next(b for b in blocks(INSTALLER) if "cases = {" in b)
        result = python_block(semantic, self.work / "release-authorization.json", self.work)
        self.assertEqual(result.returncode, 0, result.stderr)
        for case in filter(None, result.stdout.split("\0")):
            name, expected, data, inp = case.split("=")
            with self.subTest(case=name):
                self.assertEqual(self.evaluate(candidate, data, inp), expected)
        data = self.work / "cross-release.json"
        inp = self.work / "empty.json"
        data.write_text(json.dumps({"plugin": "resource", "resource-path": ["default", "image-key", "other-release"]}))
        inp.write_text("{}")
        self.assertEqual(self.evaluate(candidate, data, inp), "false")
        self.assertEqual(self.evaluate(candidate, data, self.work / "positive.input.json"), "false")
        # Each authorized resource works, not just the first sorted path.
        for path in self.paths:
            data.write_text(json.dumps({"plugin": "resource", "resource-path": path.split("/")}))
            self.assertEqual(self.evaluate(candidate, data, self.work / "positive.input.json"), "true")

    def evaluate(self, candidate, data, inp):
        result = subprocess.run(
            [
                "opa",
                "eval",
                "--format",
                "raw",
                "--data",
                str(candidate),
                "--data",
                str(data),
                "--input",
                str(inp),
                "data.policy.allow",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()


class OpaInstallationTests(unittest.TestCase):
    def run_installer(self, mode):
        # Execute the real stage body, with all privileged/network commands
        # replaced by local shell functions. Never run apt/systemctl/install.
        body = (ROOT / "service/01-install-host-tools.sh").read_text().split("require_sudo\n", 1)[1]
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ, TMPDIR=temp, TEST_MODE=mode, TEST_LOG=str(Path(temp) / "calls"))
            harness = r"""
set -Eeuo pipefail
die() { echo "$*" >&2; exit 1; }
uname() { echo x86_64; }
sudo() { printf '%s\n' "$*" >> "$TEST_LOG"; }
docker() { :; }
nginx() { :; }
openssl() { :; }
python3() { :; }
opa() { echo mock-opa; }
curl() {
    [[ "$TEST_MODE" != download-failure ]] || return 22
    local output
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == --output ]]; then output="$2"; shift; fi
        shift
    done
    if [[ "$TEST_MODE" == success ]]; then
        cp "$TEST_OPA_BINARY" "$output"
    else
        printf 'corrupt binary' > "$output"
    fi
}
"""
            result = subprocess.run(["bash", "-c", harness + body], env=env, text=True, capture_output=True)
            return result, (Path(temp) / "calls").read_text()

    def test_failed_download_does_not_install_opa(self):
        result, calls = self.run_installer("download-failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("/usr/local/bin/opa", calls)

    def test_bad_checksum_does_not_install_opa(self):
        result, calls = self.run_installer("bad-checksum")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAILED", result.stdout)
        self.assertNotIn("/usr/local/bin/opa", calls)

    @unittest.skipUnless(os.environ.get("TEST_OPA_BINARY"), "Set TEST_OPA_BINARY to the pinned official release binary")
    def test_verified_download_installs_opa(self):
        result, calls = self.run_installer("success")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("/usr/local/bin/opa", calls)


if __name__ == "__main__":
    unittest.main()

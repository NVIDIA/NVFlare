# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Local tests only: no real sudo, network, or Trustee writes."""

import base64
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SERVICE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reference_values", SERVICE / "lib/platform-reference-values.py")
parser = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parser)
VALUES = dict(zip(parser.FIELDS, ["a" * 96, 8, 0, 23, 62]))

MOCK_CLIENT = """#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[sys.argv.index("config") + 1:]
if args[:1] == ["--admin-token-file"]:
    args = args[2:]
path = pathlib.Path(os.environ["TEST_REFERENCE_STATE"])
state = json.loads(path.read_text())
with open(os.environ["TEST_REFERENCE_CALLS"], "a") as log:
    log.write(json.dumps(args) + "\\n")
if args[0] == "set-sample-reference-value":
    key, value = args[1:3]
    if key == "snp_launch_measurement":
        if os.environ.get("TEST_REFERENCE_FAIL"):
            sys.exit(42)
        state[key] = [value]
    else:
        assert "--as-integer" in args and "--as-single-value" in args
        state[key] = int(value)
    path.write_text(json.dumps(state))
elif args[0] == "get-reference-value":
    print(json.dumps(json.dumps(state[args[2]])))
elif args[0] == "set-attestation-policy":
    assert args[args.index("--id") + 1] == "default_cpu"
else:
    raise SystemExit("Unexpected administrative operation")
"""

# Shell integration uses an isolated stand-in for the authenticated HTTP write.
# Separate unit tests below exercise the actual request builder/HTTPS client.
MOCK_HTTP = """import json, os, pathlib, sys
values = json.loads(pathlib.Path(sys.argv[3]).read_text())
items = values["snp_launch_measurement"]
items = [items] if isinstance(items, str) else sorted(items)
with open(os.environ["TEST_REFERENCE_CALLS"], "a") as log:
    log.write(json.dumps(["set-measurement-list", "snp_launch_measurement", items]) + "\\n")
if os.environ.get("TEST_REFERENCE_FAIL"):
    sys.exit(42)
path = pathlib.Path(os.environ["TEST_REFERENCE_STATE"])
state = json.loads(path.read_text())
state["snp_launch_measurement"] = items
path.write_text(json.dumps(state))
"""


class ReferenceTests(unittest.TestCase):
    def test_operator_script_order_matches_documentation(self):
        expected = [
            "01-install-host-tools.sh",
            "02-install-platform-reference-values.sh",
            "03-preflight.sh",
            "04-build-trustee-main.sh",
            "05-deploy-trustee.sh",
            "06-configure-trustee-tls.sh",
            "07-harden-kbs-admin-audience.sh",
            "08-deploy-private-registry.sh",
            "09-install-platform-policy.sh",
            "10-verify-platform-reference-values.sh",
            "11-verify-service.sh",
            "12-install-trusted-service-handoff.sh",
            "13-verify-workload-release.sh",
        ]
        self.assertEqual(sorted(path.name for path in SERVICE.glob("*.sh")), expected)
        readme = (SERVICE / "README.md").read_text()
        positions = []
        for index, name in enumerate(expected[:11], start=1):
            positions.append(readme.index(f"| {index:02d} | `bash ./{name}"))
        self.assertEqual(positions, sorted(positions))
        for name in expected:
            self.assertTrue(os.access(SERVICE / name, os.X_OK), name)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / "values.json"
        self.input.write_text(json.dumps(VALUES))

    def test_exact_values(self):
        self.assertEqual(parser.load_values(self.input), VALUES)

    def test_invalid_values(self):
        bad = []
        for key in VALUES:
            candidate = VALUES.copy()
            del candidate[key]
            bad.append(candidate)
        bad.extend([dict(VALUES, extra=1), [], dict(VALUES, snp_launch_measurement="A" * 96)])
        for value in (True, False, "8", -1, 256, 8.0, None):
            bad.append(dict(VALUES, snp_min_reported_tcb_snp=value))
        for value in bad:
            with self.subTest(value=value):
                self.input.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    parser.load_values(self.input)

    def test_duplicate_keys(self):
        self.input.write_text(json.dumps(VALUES)[:-1] + ', "snp_min_reported_tcb_snp": 23}')
        with self.assertRaises(ValueError):
            parser.load_values(self.input)

    def test_measurement_allowlist_validation(self):
        for items in (["a" * 96], ["b" * 96, "a" * 96], [f"{i:096x}" for i in range(64)]):
            values = dict(VALUES, snp_launch_measurement=items)
            self.input.write_text(json.dumps(values))
            self.assertEqual(parser.load_values(self.input), values)
        for items in (
            [],
            ["a" * 96] * 2,
            ["A" * 96],
            ["a" * 95],
            [True],
            [None],
            [["a" * 96]],
            42,
            {"a": "b"},
            [f"{i:096x}" for i in range(65)],
        ):
            with self.subTest(items=items), self.assertRaises(ValueError):
                parser.validate_values(dict(VALUES, snp_launch_measurement=items))
        self.input.write_bytes(b" " * 8193)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            parser.load_values(self.input)

    def test_allowlist_exact_readback(self):
        values = dict(VALUES, snp_launch_measurement=["b" * 96, "a" * 96])
        for raw in (json.dumps(["a" * 96, "b" * 96]), json.dumps(json.dumps(["b" * 96, "a" * 96]))):
            parser.compare_reference(values, "snp_launch_measurement", raw)
        for actual in (["a" * 96], ["a" * 96, "b" * 96, "c" * 96], ["a" * 96] * 2, [], None, "a" * 96):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                parser.compare_reference(values, "snp_launch_measurement", json.dumps(actual))

    def test_allowlist_environment_roundtrip_and_return_to_single(self):
        env = self.root / "platform.env"
        env.write_text("".join(f'{v}=""\n' for v in parser.FIELDS.values()))
        for measurement in (["b" * 96, "a" * 96], "a" * 96):
            values = dict(VALUES, snp_launch_measurement=measurement)
            parser.update_env(values, env)
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; printf "%s\\n" ' + " ".join(f'"${v}"' for v in parser.FIELDS.values()),
                    "test",
                    str(env),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            restored = parser.from_environment_args(result.stdout.splitlines())
            self.assertEqual(parser.measurements(restored["snp_launch_measurement"]), parser.measurements(measurement))
            self.assertEqual(list(restored.values())[1:], list(VALUES.values())[1:])

    def test_native_array_https_request_and_no_redirects(self):
        values = dict(VALUES, snp_launch_measurement=["b" * 96, "a" * 96])
        token = self.root / "token"
        token.write_text("dummy-admin-token\n")
        for status in (200, 302, 500):
            with (
                patch.object(parser.ssl, "create_default_context") as context,
                patch.object(parser.http.client, "HTTPSConnection") as connect,
            ):
                connection = connect.return_value
                connection.getresponse.return_value.status = status
                if status == 200:
                    parser.install_measurements(values, "https://example.invalid:8443", "ca.crt", token)
                else:
                    with self.assertRaisesRegex(ValueError, f"HTTP {status}"):
                        parser.install_measurements(values, "https://example.invalid:8443", "ca.crt", token)
                context.assert_called_once_with(cafile="ca.crt")
                connect.assert_called_once_with("example.invalid", 8443, context=context.return_value, timeout=30)
                self.assertEqual(context.return_value.minimum_version, parser.ssl.TLSVersion.TLSv1_2)
                args, kwargs = connection.request.call_args
                self.assertEqual(args, ("POST", "/kbs/v0/reference-value"))
                self.assertEqual(kwargs["headers"]["Authorization"], "Bearer dummy-admin-token")
                message = json.loads(kwargs["body"])
                self.assertEqual(message["type"], "sample")
                self.assertEqual(message["version"], "0.1.0")
                self.assertEqual(
                    json.loads(base64.b64decode(message["payload"])), {"snp_launch_measurement": ["a" * 96, "b" * 96]}
                )
                connection.request.assert_called_once()
                connection.close.assert_called_once()

    def test_http_rejects_unsafe_origin_or_token(self):
        token = self.root / "token"
        token.write_text("dummy")
        for url in (
            "http://example.invalid",
            "https://user:pass@example.invalid",
            "https://example.invalid/path",
            "https://example.invalid?query=1",
            "https://example.invalid#fragment",
        ):
            with self.subTest(url=url), patch.object(parser.http.client, "HTTPSConnection") as connect:
                with self.assertRaises(ValueError):
                    parser.install_measurements(VALUES, url, "ca.crt", token)
                connect.assert_not_called()
        for invalid in ("", "contains\nnewline", "x" * 16385):
            token.write_text(invalid)
            with self.assertRaisesRegex(ValueError, "token"):
                parser.install_measurements(VALUES, "https://example.invalid", "ca.crt", token)

    def test_tls_or_connection_error_is_not_retried(self):
        token = self.root / "token"
        token.write_text("dummy")
        with (
            patch.object(parser.ssl, "create_default_context"),
            patch.object(parser.http.client, "HTTPSConnection") as connect,
        ):
            connection = connect.return_value
            connection.request.side_effect = parser.ssl.SSLCertVerificationError("untrusted certificate")
            with self.assertRaises(parser.ssl.SSLCertVerificationError):
                parser.install_measurements(VALUES, "https://example.invalid", "ca.crt", token)
            connection.request.assert_called_once()
            connection.close.assert_called_once()

    def test_environment_preserves_other_fields(self):
        env = self.root / "platform.env"
        env.write_text('KEEP="unchanged"\n' + "".join(f'{v}=""\n' for v in parser.FIELDS.values()))
        parser.update_env(VALUES, env)
        self.assertIn('KEEP="unchanged"\n', env.read_text())
        self.assertEqual(env.stat().st_mode & 0o777, 0o600)
        before = env.read_bytes()
        env.write_bytes(before + b'SNP_LAUNCH_MEASUREMENT="duplicate"\n')
        malformed = env.read_bytes()
        with self.assertRaises(ValueError):
            parser.update_env(VALUES, env)
        self.assertEqual(env.read_bytes(), malformed)

    def test_live_reference_decoding(self):
        for key, value in VALUES.items():
            expected = [value] if key == "snp_launch_measurement" else value
            for raw in (json.dumps(expected), json.dumps(json.dumps(expected))):
                parser.compare_reference(VALUES, key, raw)
        for raw in ("null", "true", '"\\"8\\""', "9", "[8]"):
            with self.assertRaises(ValueError):
                parser.compare_reference(VALUES, "snp_min_reported_tcb_bootloader", raw)

    def fixture(self):
        kit = self.root / "kit"
        shutil.copytree(SERVICE, kit, ignore=shutil.ignore_patterns("tests", "__pycache__"))
        backend = self.root / "backend"
        for part in (
            "kbs/config/docker-compose",
            "kbs/data/kbs-policy",
            "kbs/data/attestation-service/attestation_service_policy",
        ):
            (backend / part).mkdir(parents=True)
        (backend / "kbs/config/docker-compose/admin-token").write_text("dummy")
        (backend / "kbs/config/docker-compose/admin-token").chmod(0o600)
        (backend / "kbs/data/kbs-policy/resource-policy.rego").write_text("default allow := false\n")
        policies = backend / "kbs/data/attestation-service/attestation_service_policy"
        shutil.copyfile(kit / "policies/default_cpu.rego", policies / "default_cpu.rego")
        (policies / "default_gpu.rego").write_text("unchanged GPU policy")
        client = backend / "kbs-client-snp-tdx-test"
        client.write_text(MOCK_CLIENT)
        client.chmod(0o755)
        cert = self.root / "public.crt"
        cert.write_text("dummy")
        host = subprocess.check_output(["hostname", "-f"], text=True).strip()
        (kit / "platform.env").write_text(
            f'EXPECTED_HOSTNAME="{host}"\nSERVICE_FQDN="secure.unit.local"\n'
            f'PUBLISHER_DIR="{self.root}/publisher"\nREGISTRY_PORT=5000\nREGISTRY_BACKEND_PORT=5001\n'
            f'TRUSTEE_ROOT="{backend}"\nTRUSTEE_LABEL="test"\nTRUSTEE_PROJECT="test"\n'
            f'KBS_URL="https://secure.unit.local:8443"\nTRUSTEE_PUBLIC_CERT="{cert}"\n'
            + "".join(f'{v}=""\n' for v in parser.FIELDS.values())
        )
        binary = self.root / "bin"
        binary.mkdir()
        sudo = binary / "sudo"
        sudo.write_text('#!/bin/sh\nif [ "$1" = -n ]; then shift; fi\nexec "$@"\n')
        sudo.chmod(0o755)
        mock_http = self.root / "mock-http.py"
        mock_http.write_text(MOCK_HTTP)
        python = binary / "python3"
        python.write_text(
            f'#!/bin/sh\nif [ "$2" = install-measurements ]; then\n'
            f'  exec "{sys.executable}" "{mock_http}" "$@"\nfi\n'
            f'exec "{sys.executable}" "$@"\n'
        )
        python.chmod(0o755)
        curl = binary / "curl"
        curl.write_text("#!/bin/sh\nexit 0\n")
        curl.chmod(0o755)
        state = self.root / "state.json"
        state.write_text(json.dumps({k: None for k in VALUES}))
        calls = self.root / "calls.jsonl"
        environment = dict(
            os.environ,
            PATH=f'{binary}:{os.environ["PATH"]}',
            TEST_REFERENCE_STATE=str(state),
            TEST_REFERENCE_CALLS=str(calls),
        )
        return kit, state, calls, environment

    def run_script(self, kit, environment, script, *args):
        return subprocess.run(
            ["bash", str(kit / script), str(self.input), *args], env=environment, capture_output=True, text=True
        )

    def test_install_and_readback(self):
        kit, state, calls, environment = self.fixture()
        result = self.run_script(
            kit, environment, "02-install-platform-reference-values.sh", "--approve-platform-reference-values"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected = dict(VALUES, snp_launch_measurement=[VALUES["snp_launch_measurement"]])
        self.assertEqual(json.loads(state.read_text()), expected)
        commands = [json.loads(x) for x in calls.read_text().splitlines()]
        self.assertEqual([x[2] for x in commands[:4]], ["255"] * 4)
        self.assertEqual(commands[4][1], "snp_launch_measurement")
        self.assertEqual(len([x for x in commands if x[0].startswith("set-")]), 9)
        result = self.run_script(kit, environment, "10-verify-platform-reference-values.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("PASS "), 5)

    def test_configure_only_has_no_backend_calls(self):
        kit, state, calls, environment = self.fixture()
        before = state.read_bytes()
        result = self.run_script(
            kit,
            environment,
            "02-install-platform-reference-values.sh",
            "--approve-platform-reference-values",
            "--configure-only",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(calls.exists())
        self.assertEqual(state.read_bytes(), before)

    def test_allowlist_install_replaces_entire_list(self):
        kit, state, calls, environment = self.fixture()
        for items in (["b" * 96, "a" * 96], ["c" * 96]):
            self.input.write_text(json.dumps(dict(VALUES, snp_launch_measurement=items)))
            result = self.run_script(
                kit, environment, "02-install-platform-reference-values.sh", "--approve-platform-reference-values"
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(state.read_text()), dict(VALUES, snp_launch_measurement=sorted(items)))

    def test_fresh_policy_installer_accepts_allowlist(self):
        kit, state, calls, environment = self.fixture()
        values = dict(VALUES, snp_launch_measurement=["b" * 96, "a" * 96])
        parser.update_env(values, kit / "platform.env")
        result = subprocess.run(
            ["bash", str(kit / "09-install-platform-policy.sh"), "--approve-pinned-snp-platform"],
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(state.read_text()), dict(VALUES, snp_launch_measurement=["a" * 96, "b" * 96]))

    def test_allowlist_configure_only_does_not_write_backend(self):
        kit, state, calls, environment = self.fixture()
        self.input.write_text(json.dumps(dict(VALUES, snp_launch_measurement=["a" * 96, "b" * 96])))
        result = self.run_script(
            kit,
            environment,
            "02-install-platform-reference-values.sh",
            "--approve-platform-reference-values",
            "--configure-only",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(calls.exists())
        self.assertIn("SNP_LAUNCH_MEASUREMENT='[", (kit / "platform.env").read_text())

    def test_concurrent_installer_rejected_before_mutation(self):
        kit, state, calls, environment = self.fixture()
        before = (kit / "platform.env").read_bytes()
        with (kit / ".platform-reference-update.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_script(
                kit, environment, "02-install-platform-reference-values.sh", "--approve-platform-reference-values"
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Another platform-reference update", result.stderr)
        self.assertEqual((kit / "platform.env").read_bytes(), before)
        self.assertFalse(calls.exists())

    def test_invalid_input_has_no_mutations(self):
        kit, state, calls, environment = self.fixture()
        self.input.write_text(json.dumps(dict(VALUES, extra="reject")))
        before = (kit / "platform.env").read_bytes()
        result = self.run_script(
            kit, environment, "02-install-platform-reference-values.sh", "--approve-platform-reference-values"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((kit / "platform.env").read_bytes(), before)
        self.assertFalse(calls.exists())

    def test_duplicate_allowlist_has_no_mutations(self):
        kit, state, calls, environment = self.fixture()
        self.input.write_text(json.dumps(dict(VALUES, snp_launch_measurement=["a" * 96] * 2)))
        before = (kit / "platform.env").read_bytes()
        result = self.run_script(
            kit, environment, "02-install-platform-reference-values.sh", "--approve-platform-reference-values"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((kit / "platform.env").read_bytes(), before)
        self.assertFalse(calls.exists())

    def test_failed_write_leaves_restrictive_staging(self):
        kit, state, calls, environment = self.fixture()
        environment["TEST_REFERENCE_FAIL"] = "1"
        result = self.run_script(
            kit, environment, "02-install-platform-reference-values.sh", "--approve-platform-reference-values"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No automatic rollback", result.stderr)
        self.assertEqual([json.loads(state.read_text())[k] for k in list(VALUES)[1:]], [255] * 4)

    def test_wrong_cpu_policy_stops_before_mutation(self):
        kit, state, calls, environment = self.fixture()
        policy = self.root / "backend/kbs/data/attestation-service/attestation_service_policy/default_cpu.rego"
        policy.write_text("unreviewed policy")
        before = (kit / "platform.env").read_bytes()
        result = self.run_script(
            kit, environment, "02-install-platform-reference-values.sh", "--approve-platform-reference-values"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((kit / "platform.env").read_bytes(), before)
        self.assertFalse(calls.exists())

    def test_readback_mismatch_is_read_only(self):
        kit, state, calls, environment = self.fixture()
        before = state.read_bytes()
        result = self.run_script(kit, environment, "10-verify-platform-reference-values.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(state.read_bytes(), before)
        self.assertTrue(all(json.loads(x)[0] == "get-reference-value" for x in calls.read_text().splitlines()))

    def test_export_contains_only_five_values(self):
        profile = self.root / "profile"
        profile.mkdir()
        config = profile / "platform-reference.final.env"
        config.write_text(
            f'PLATFORM_WORK_ROOT="{self.root}"\nPLATFORM_PROFILE="profile"\n'
            + f'APPROVED_SNP_LAUNCH_MEASUREMENT="{VALUES["snp_launch_measurement"]}"\n'
            + "".join(f'{parser.FIELDS[k]}="{VALUES[k]}"\n' for k in list(VALUES)[1:])
            + 'PLATFORM_REFERENCE_SIGNING_KEY="private-not-exported"\n'
        )
        output = self.root / "export.json"
        command = [
            "bash",
            str(SERVICE.parent / "trusted_system/10-export-platform-reference-values.sh"),
            str(config),
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text()), VALUES)
        self.assertNotIn("private", output.read_text())
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()

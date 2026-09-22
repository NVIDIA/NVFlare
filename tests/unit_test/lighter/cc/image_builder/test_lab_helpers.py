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

"""Lab profile selection and diagnostics with application-service confinement."""

import contextlib
import http.client
import importlib.util
import io
import json
import os
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from cvm.build import config
from cvm.common.errors import BuildError
from cvm.common.io import read_json, write_json
from cvm.common.references import SNP_BOOLS, SNP_INTS, SNP_LISTS, TDX_HEX
from cvm.runtime import bootstrap

LAB = Path(__file__).resolve().parents[4] / "integration_test/lighter/cc/image_builder"


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, LAB / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LabProfileTests(unittest.TestCase):
    def test_selected_profiles_require_only_their_own_references(self):
        prepare = load_helper("prepare_lab")
        for selected in (None, ["intel_tdx"], ["amd_sev_snp"]):
            expected = set(selected or ("intel_tdx", "amd_sev_snp"))
            with self.subTest(platforms=selected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                firmware = root / "tdx.fd"
                if "intel_tdx" in expected:
                    firmware.touch()
                with (
                    patch.dict(os.environ, CVM_TDX_FIRMWARE=str(firmware)),
                    patch.object(prepare.subprocess, "check_output", return_value="Candidate: 1.0\n"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    prepare.prepare(root, platforms=selected)
                path = root / "inputs/lab-profile.yml"
                refs_path = root / "inputs/test-tcb-references.json"
                self.assertEqual(read_json(refs_path), {})
                refs = {}
                if "intel_tdx" in expected:
                    refs.update({name: ["0" * size] for name, size in TDX_HEX.items()})
                    refs["allowed_advisory_ids"] = []
                if "amd_sev_snp" in expected:
                    refs.update({name: [1] for name in SNP_LISTS})
                    refs.update({name: False for name in SNP_BOOLS})
                    refs.update({name: 0 for name in SNP_INTS})
                # Keep profile parsing/reference validation real; the test does
                # not build a guest or claim these synthetic values are approved.
                with (
                    patch.object(config, "local_path", side_effect=lambda path, value: value),
                    patch.object(config, "validate_kbs_client_provenance"),
                ):
                    with self.assertRaisesRegex(BuildError, "Missing approved TCB"):
                        config.profile(path)
                    write_json(refs_path, refs)
                    profile = config.profile(path)
                self.assertEqual(set(profile["platforms"]), expected)

    def test_only_selected_tdx_requires_tdvf(self):
        prepare = load_helper("prepare_lab")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, CVM_TDX_FIRMWARE=str(root / "missing.fd")):
                with self.assertRaisesRegex(SystemExit, "validated TDVF"):
                    prepare.prepare(root, platforms=["intel_tdx"])
            self.assertFalse(list(root.glob("lab-pki-*")))

    def test_http_only_needs_no_platform_inputs(self):
        prepare = load_helper("prepare_lab")
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.dict(os.environ, CVM_TDX_FIRMWARE=str(Path(directory) / "missing.fd")),
                patch.object(prepare.subprocess, "check_output") as apt,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                prepare.prepare(directory, http_only=True)
            apt.assert_not_called()
            self.assertEqual(set(read_json(Path(directory) / "lab-state.json")), {"pki"})

    def test_cli_supports_one_or_repeated_platform_selection(self):
        prepare = load_helper("prepare_lab")
        for selected in (["intel_tdx"], ["amd_sev_snp"], ["intel_tdx", "amd_sev_snp"]):
            argv = ["prepare_lab.py", "/lab"] + [arg for platform in selected for arg in ("-p", platform)]
            with self.subTest(platforms=selected), patch("sys.argv", argv), patch.object(prepare, "prepare") as call:
                prepare.main()
            call.assert_called_once_with("/lab", http_only=False, platforms=selected)

    def test_invalid_platform_selection_is_rejected_before_creating_inputs(self):
        prepare = load_helper("prepare_lab")
        for selected in ([], ["unknown"], ["intel_tdx", "intel_tdx"]):
            with self.subTest(platforms=selected), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(SystemExit):
                    prepare.prepare(directory, platforms=selected)
                self.assertEqual(list(Path(directory).iterdir()), [])


class LabStateTests(unittest.TestCase):
    def test_bootstrap_publishes_only_successfully_verified_firewall_status(self):
        for failure in (None, "apply", "verify"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                status = Path(directory) / "firewall.json"
                write_json(status, {"present": True, "verified_at": 1})

                def execute(argv, **kwargs):
                    self.assertFalse(status.exists(), "Old verification must not survive a replacement attempt")
                    if (failure == "apply" and "-f" in argv) or (failure == "verify" and "list" in argv):
                        raise BuildError("nft failed")
                    return b"table inet cvm {}"

                with (
                    patch.object(bootstrap, "STATE", Path(directory)),
                    patch.object(bootstrap, "run", side_effect=execute) as run,
                    patch.object(bootstrap.time, "time", return_value=123),
                ):
                    if failure:
                        with self.assertRaises(BuildError):
                            bootstrap.firewall([], [443])
                        self.assertFalse(status.exists())
                    else:
                        bootstrap.firewall([], [443])
                        self.assertEqual(read_json(status), {"present": True, "verified_at": 123})
                        self.assertEqual(status.stat().st_mode & 0o777, 0o444)
                        self.assertEqual(run.call_args.args[0], ["nft", "list", "table", "inet", "cvm"])

    def test_agent_consumes_bootstrap_status_without_privileged_commands(self):
        agent = load_helper("lab_guest_agent")
        with patch.object(agent, "read_json", return_value={"present": True, "verified_at": 123}) as read:
            with patch.object(agent, "run", side_effect=AssertionError("No privileged command allowed")):
                self.assertEqual(agent.firewall_state(), {"firewall_present": True, "firewall_verified_at": 123})
            read.assert_called_once_with("/run/cvm/firewall.json")
        with patch.object(agent, "read_json", return_value={"present": False}):
            with self.assertRaisesRegex(BuildError, "No verified bootstrap firewall"):
                agent.firewall_state()
        with patch.object(agent, "read_json", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(BuildError, "Bootstrap firewall verification is unavailable"):
                agent.firewall_state()

    def test_state_errors_are_json_http_diagnostics_and_server_recovers(self):
        agent = load_helper("lab_guest_agent")
        server = agent.http.server.HTTPServer(("127.0.0.1", 0), agent.Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            for error in (BuildError("No verified bootstrap firewall status"), OSError("SECRET_SENTINEL")):
                with self.subTest(error=type(error).__name__), patch.object(agent, "state", side_effect=error):
                    connection = http.client.HTTPConnection(*server.server_address, timeout=3)
                    try:
                        connection.request("GET", "/state")
                        response = connection.getresponse()
                        self.assertEqual(response.status, 500)
                        self.assertEqual(response.getheader("Content-Type"), "application/json")
                        body = response.read()
                        self.assertNotIn(b"SECRET_SENTINEL", body)
                        detail = json.loads(body)
                        self.assertEqual(detail["state_error"], type(error).__name__)
                        if isinstance(error, BuildError):
                            self.assertEqual(detail["diagnostic"], str(error))
                    finally:
                        connection.close()
            with patch.object(agent, "state", return_value={"firewall_present": True}):
                connection = http.client.HTTPConnection(*server.server_address, timeout=3)
                try:
                    connection.request("GET", "/state")
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertEqual(json.loads(response.read()), {"firewall_present": True})
                finally:
                    connection.close()
        finally:
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()

    def test_agent_accepts_the_exact_production_named_candidate_profile(self):
        agent = load_helper("lab_guest_agent")
        with (
            patch.object(agent, "read_json", return_value={"profile_version": "production"}),
            patch.object(agent, "protect_process") as protect,
            patch.object(agent.http.server, "HTTPServer") as server,
            patch.object(agent.sys, "argv", [agent.__file__]),
        ):
            agent.main()
        protect.assert_called_once_with()
        server.assert_called_once_with(("0.0.0.0", 18081), agent.Handler)
        server.return_value.__enter__.return_value.serve_forever.assert_called_once_with()

    def test_hardware_runner_reports_http_diagnostic_instead_of_readiness_timeout(self):
        hardware = load_helper("test_hardware")
        case = hardware.HardwareTests()
        case.process = Mock()
        case.process.poll.return_value = None
        case.request = Mock(
            side_effect=urllib.error.HTTPError(
                "http://127.0.0.1/state", 500, "failure", {}, io.BytesIO(b'{"diagnostic":"No verified firewall"}')
            )
        )
        with self.assertRaisesRegex(AssertionError, "HTTP 500.*No verified firewall"):
            case.ready()
        case.request.assert_called_once()

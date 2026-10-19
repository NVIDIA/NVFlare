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
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

from cvm.artifacts.bundle import required_acceptance_checks
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
    def test_lab_kbs_adds_profile_nvidia_remote_verifier(self):
        lab_kbs = load_helper("lab_kbs")
        base = {"dcap_verifier": {"tcb_update_type": "standard"}}
        profiles = [
            {
                "gpu": "nvidia_cc",
                "gpu_attestation_url": "https://nras.example.org/v4/attest/gpu/",
            }
        ]
        with patch.object(
            lab_kbs,
            "read_json",
            return_value={"attestation_service": {"verifier_config": base}},
        ):
            result = lab_kbs.verifier_config(profiles)

        self.assertEqual(result["dcap_verifier"], {"tcb_update_type": "standard"})
        self.assertEqual(
            result["nvidia_verifier"],
            {"type": "Remote", "verifier_url": "https://nras.example.org/v4/attest"},
        )

    def test_cpu_lab_kbs_does_not_add_nvidia_verifier(self):
        lab_kbs = load_helper("lab_kbs")
        base = {"dcap_verifier": {"tcb_update_type": "standard"}}
        with patch.object(
            lab_kbs,
            "read_json",
            return_value={"attestation_service": {"verifier_config": base}},
        ):
            result = lab_kbs.verifier_config([{"gpu": "none"}])

        self.assertEqual(result, {"dcap_verifier": {"tcb_update_type": "standard"}})
        self.assertNotIn("nvidia_verifier", result)

    def test_lab_kbs_rejects_conflicting_nvidia_verifier_urls(self):
        lab_kbs = load_helper("lab_kbs")
        profiles = [
            {"gpu": "nvidia_cc", "gpu_attestation_url": "https://first.example.org/v4/attest/gpu"},
            {"gpu": "nvidia_cc", "gpu_attestation_url": "https://second.example.org/v4/attest/gpu"},
        ]
        with (
            patch.object(
                lab_kbs,
                "read_json",
                return_value={"attestation_service": {"verifier_config": {}}},
            ),
            self.assertRaisesRegex(BuildError, "must use the same NVIDIA verifier URL"),
        ):
            lab_kbs.verifier_config(profiles)

    def test_lab_kbs_rejects_invalid_nvidia_verifier_endpoints(self):
        lab_kbs = load_helper("lab_kbs")
        invalid = (
            "https://nras.example.org",
            "https://nras.example.org/v4/attest",
            "https://nras.example.org/v4/attest/gpu?tenant=lab",
            "https://nras.example.org/v4/attest/gpu#fragment",
        )
        for endpoint in invalid:
            profiles = [{"gpu": "nvidia_cc", "gpu_attestation_url": endpoint}]
            with (
                self.subTest(endpoint=endpoint),
                patch.object(
                    lab_kbs,
                    "read_json",
                    return_value={"attestation_service": {"verifier_config": {}}},
                ),
                self.assertRaisesRegex(BuildError, "must end in /gpu"),
            ):
                lab_kbs.verifier_config(profiles)

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
            call.assert_called_once_with(
                "/lab", http_only=False, platforms=selected, package_profile=None, pins_from_host=False
            )

    def test_cli_forwards_package_pin_overrides(self):
        """Pins come from the checked-in profile unless the operator opts into another source."""
        prepare = load_helper("prepare_lab")
        argv = ["prepare_lab.py", "/lab", "--package-profile", "pins.yml", "--pins-from-host"]
        with patch("sys.argv", argv), patch.object(prepare, "prepare") as call:
            prepare.main()
        call.assert_called_once_with(
            "/lab", http_only=False, platforms=None, package_profile="pins.yml", pins_from_host=True
        )

    def test_invalid_platform_selection_is_rejected_before_creating_inputs(self):
        prepare = load_helper("prepare_lab")
        for selected in ([], ["unknown"], ["intel_tdx", "intel_tdx"]):
            with self.subTest(platforms=selected), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(SystemExit):
                    prepare.prepare(directory, platforms=selected)
                self.assertEqual(list(Path(directory).iterdir()), [])


class LabStateTests(unittest.TestCase):
    def test_official_suites_generate_every_required_acceptance_check(self):
        hardware = load_helper("test_hardware")
        trustee = load_helper("test_http")
        suites = (hardware, trustee)
        for suite in suites:
            methods = {name for name in dir(suite.HttpTests if suite is trustee else suite.HardwareTests)}
            self.assertLessEqual(set(suite.ACCEPTANCE_CHECKS), methods)
        for platform, gpu in (
            ("intel_tdx", "none"),
            ("amd_sev_snp", "none"),
            ("intel_tdx", "nvidia_cc"),
            ("amd_sev_snp", "nvidia_cc"),
        ):
            for vault_prescan in (True, False):
                manifest = {"platform": platform, "contract": {"gpu": gpu, "vault_prescan": vault_prescan}}
                required = required_acceptance_checks(manifest)
                generated = Counter(
                    check
                    for checks in (
                        *hardware.applicable_acceptance_checks(manifest).values(),
                        *trustee.ACCEPTANCE_CHECKS.values(),
                    )
                    for check in checks
                )
                with self.subTest(platform=platform, gpu=gpu, vault_prescan=vault_prescan):
                    self.assertEqual("attestation_quarantine_recovery" in required, gpu == "none")
                    self.assertEqual("periodic_gpu_denial" in required, gpu == "nvidia_cc")
                    self.assertEqual(set(generated) & required, required)
                    self.assertTrue(all(generated[check] == 1 for check in required))

    def test_acceptance_runner_requires_all_external_inputs(self):
        runner = load_helper("acceptance_runner")
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(BuildError, "CVM_ACCEPTANCE_VAULT"):
            runner.required_environment()
        values = {
            "CVM_ACCEPTANCE_VAULT": "/vault",
            "CVM_ACCEPTANCE_RESULT_KEY": "/evidence.key",
            "CVM_LAB_DIRECTORY": "/lab",
            "CVM_OTHER_RESOURCE": "keys/other/binding",
        }
        with patch.dict(os.environ, values, clear=True):
            self.assertEqual(runner.required_environment(), values)

    def test_gpu_retrievals_include_composite_claims_unless_cpu_only_is_explicit(self):
        trustee = load_helper("test_http")
        case = trustee.HttpTests()
        case.manifest = {
            "attestation_policy_id": "gpu-policy",
            "contract": {"gpu": "nvidia_cc", "gpu_count": 2},
        }
        claims = case.retrieval_claims({})
        self.assertEqual(set(claims["gpu_submods"]), {"gpu0", "gpu1"})
        self.assertTrue(
            all(submod["ear.appraisal-policy-id"] == "gpu-policy" for submod in claims["gpu_submods"].values())
        )
        self.assertEqual(case.retrieval_claims({"gpu_submods": {}}), {"gpu_submods": {}})

    def test_failed_acceptance_case_cannot_reach_the_evidence_signer(self):
        trustee = load_helper("test_http")
        case = trustee.HttpTests("test_negative_wrong_policy_and_expired_ear_denied")
        case.evidence_enabled = True
        case.create = Mock(return_value=(b"digest", "keys/build/binding", b"secret"))
        case.retrieve = Mock(side_effect=(BuildError("denied"), None, BuildError("denied")))
        case.result = Mock()
        with self.assertRaises(AssertionError):
            case.test_negative_wrong_policy_and_expired_ear_denied()
        case.result.assert_not_called()

    def test_acceptance_runner_executes_both_suites_and_selects_gpu_tests(self):
        runner = load_helper("acceptance_runner")
        for gpu in ("none", "nvidia_cc"):
            with self.subTest(gpu=gpu), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = root / "bundle"
                vault = root / "vault"
                lab = root / "lab"
                output = root / "evidence"
                report = root / "acceptance-report.json"
                bundle.mkdir()
                vault.mkdir()
                (lab / "lab-kbs").mkdir(parents=True)
                (lab / "lab-kbs/admin.json").touch()
                environment = {
                    "CVM_ACCEPTANCE_VAULT": str(vault),
                    "CVM_ACCEPTANCE_RESULT_KEY": str(root / "evidence.key"),
                    "CVM_LAB_DIRECTORY": str(lab),
                    "CVM_OTHER_RESOURCE": "keys/other/binding",
                    "CVM_ACCEPTANCE_OUTPUT": str(output),
                }
                signing_key = Mock()
                public_key = signing_key.public_key.return_value
                aggregate = {"schema_version": 1, "checks": {}}
                manifest = {"production_ready": True, "contract": {"gpu": gpu}}
                with (
                    patch.dict(os.environ, environment, clear=True),
                    patch.object(runner, "verify_bundle", return_value=manifest),
                    patch.object(runner, "load_signing_key", return_value=signing_key),
                    patch.object(runner, "run_suite") as run_suite,
                    patch.object(runner, "aggregate", return_value=aggregate) as aggregate_results,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    runner.main([bundle, report])
                self.assertEqual(
                    [call.args[0] for call in run_suite.call_args_list], ["test_http.py", "test_hardware.py"]
                )
                suite_environment = run_suite.call_args_list[0].args[1]
                self.assertEqual(suite_environment["CVM_BUNDLE"], str(bundle.resolve()))
                self.assertEqual(suite_environment["CVM_VAULT"], str(vault.resolve()))
                self.assertEqual(suite_environment.get("CVM_GPU_HARDWARE_TESTS"), "1" if gpu == "nvidia_cc" else None)
                aggregate_results.assert_called_once_with(bundle.resolve(), [output.resolve()], [public_key])
                self.assertEqual(read_json(report), aggregate)

    def test_acceptance_runner_interrupts_suite_group_and_keeps_streamed_log(self):
        runner = load_helper("acceptance_runner")
        process = Mock(pid=1234)
        process.poll.side_effect = [None, None, 0]
        process.wait.side_effect = [runner._SuiteSignal(runner.signal.SIGTERM), 0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (
                patch.object(runner.subprocess, "Popen", return_value=process) as popen,
                patch.object(runner.signal, "signal", return_value=runner.signal.SIG_DFL),
                patch.object(runner.os, "killpg") as killpg,
                self.assertRaisesRegex(BuildError, "interrupted by SIGTERM.*bounded grace period"),
            ):
                runner.run_suite("test_http.py", {}, output)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertIn("--catch", popen.call_args.args[0])
        self.assertIs(popen.call_args.kwargs["stderr"], runner.subprocess.STDOUT)
        self.assertNotEqual(popen.call_args.kwargs["stdout"], runner.subprocess.PIPE)
        killpg.assert_called_once_with(1234, runner.signal.SIGINT)
        process.wait.assert_called_with(timeout=runner.SUITE_SHUTDOWN_SECONDS)

    def test_acceptance_runner_stops_qemu_left_without_its_launcher(self):
        runner = load_helper("acceptance_runner")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            delivery = output / "case/delivery"
            delivery.mkdir(parents=True)
            (delivery / "vault.qcow2").touch()
            state_path = output / "runtime.json"
            write_json(
                state_path,
                {
                    "vault_directory": str(delivery.resolve()),
                    "launcher_pid": 100,
                    "launcher_start": 1,
                    "qemu_pid": 200,
                    "qemu_start": 2,
                },
            )
            with (
                patch.object(runner, "runtime_state_path", return_value=state_path),
                patch.object(runner, "shutdown", side_effect=BuildError("launcher exited")),
                patch.object(runner, "_process_start", side_effect=[2, 2, FileNotFoundError]),
                patch.object(runner, "qmp_powerdown") as powerdown,
                patch.object(runner.time, "sleep"),
                patch.object(runner, "require_detached") as detached,
            ):
                runner._shutdown_vaults(output)
            powerdown.assert_called_once_with(runner.qmp_socket_path(delivery))
            detached.assert_called_once_with(delivery)
            self.assertFalse(state_path.exists())

    def test_cross_vault_attestation_uses_profile_policy_selector(self):
        agent = load_helper("lab_guest_agent")
        agent.CONFIG = {
            "kbs_client": "/usr/bin/kbs-client",
            "kbs_url": "https://kbs.example.org",
            "kbs_cert": "/etc/cvm/kbs.pem",
            "gpu": "none",
            "attestation_policy_selector": "cvm-cpu-profile",
        }
        with (
            patch.object(agent, "validate_resource"),
            patch.object(agent, "read_json", return_value={"digest": "ab" * 32}),
            patch.object(agent, "_fresh_credentials", side_effect=BuildError("stop after attestation")) as fresh,
            self.assertRaisesRegex(BuildError, "stop after attestation"),
        ):
            agent.cross_vault("keys/other/resource")

        self.assertIs(fresh.call_args.args[0], agent.CONFIG)
        self.assertEqual(fresh.call_args.args[1], bytes.fromhex("ab" * 32))
        self.assertGreater(fresh.call_args.args[2](), 0)

    def test_cross_vault_reports_missing_policy_selector(self):
        agent = load_helper("lab_guest_agent")
        agent.CONFIG = {"gpu": "none"}
        with (
            patch.object(agent, "validate_resource"),
            patch.object(agent, "read_json", side_effect=AssertionError("binding must not be read")),
            self.assertRaisesRegex(BuildError, "runtime.json lacks attestation_policy_selector"),
        ):
            agent.cross_vault("keys/other/resource")

    def test_cross_vault_handler_reports_only_safe_diagnostics(self):
        agent = load_helper("lab_guest_agent")
        server = agent.http.server.HTTPServer(("127.0.0.1", 0), agent.Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            for error, diagnostic in (
                (BuildError("Wrong appraisal policy"), "Wrong appraisal policy"),
                (OSError("SECRET_SENTINEL"), "Test fixture exception"),
            ):
                with self.subTest(error=type(error).__name__), patch.object(agent, "cross_vault", side_effect=error):
                    connection = http.client.HTTPConnection(*server.server_address, timeout=3)
                    try:
                        connection.request("POST", "/cross-vault?resource=keys/other/resource")
                        response = connection.getresponse()
                        self.assertEqual(response.status, 200)
                        detail = json.loads(response.read())
                        self.assertEqual(detail["fault_error"], type(error).__name__)
                        self.assertEqual(detail["diagnostic"], diagnostic)
                        self.assertNotIn("SECRET_SENTINEL", json.dumps(detail))
                    finally:
                        connection.close()
        finally:
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()

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

    def test_gpu_denial_accepts_stable_poweroff_without_a_final_bootstrap_message(self):
        hardware = load_helper("test_hardware")
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "boot.log"
            log.write_text("systemd: Forcibly powering off\nPower down\n")
            case = hardware.HardwareTests()
            case.logs = [log]
            case.assert_powered_off()

    def test_hardware_acceptance_allows_an_exact_production_named_profile(self):
        hardware = load_helper("test_hardware")
        manifest = {"profile_version": "cpu-production-r1", "dev_mode": False}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {
                "CVM_BUNDLE": str(root / "bundle"),
                "CVM_VAULT": str(root / "vault"),
                "CVM_HARDWARE_OUTPUT": str(root / "evidence"),
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.object(hardware, "verify_bundle", return_value=manifest),
                patch.object(hardware.os, "geteuid", return_value=0),
            ):
                hardware.HardwareTests.setUpClass()
        self.assertEqual(hardware.HardwareTests.manifest, manifest)

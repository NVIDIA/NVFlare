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

"""Public command routing, failure status and secret-free vault diagnostics."""

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from cvm import __main__ as cli
from cvm.common.errors import BuildError


class CliTests(unittest.TestCase):
    def test_vault_configuration_diagnostics_never_echo_input_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "vault_build.yml"
            (root / "application.tar").touch()
            valid = {
                "image_id": "sha256:" + "a" * 64,
                "docker_archive": "application.tar",
                "cvm_image": str(root),
            }
            cases = (
                ("container: [secret-token\n", "Invalid configuration YAML at line"),
                ("secret-token: unexpected\n", "Unknown application input"),
                (dict(valid, requires_gpu="secret-token"), "requires_gpu must be boolean"),
                (dict(valid, docker_archive="secret-token.tar"), "Invalid vault_build.yml inputs"),
            )
            for value, diagnostic in cases:
                path.write_text(value if isinstance(value, str) else json.dumps(value))
                stderr = io.StringIO()
                with (
                    self.subTest(diagnostic=diagnostic),
                    patch.object(cli.vault, "protect_process") as protect,
                    patch.object(cli.vault, "profile_from_image") as profile,
                    contextlib.redirect_stderr(stderr),
                    self.assertRaises(SystemExit) as error,
                ):
                    cli.main(["vault", str(path)])
                self.assertEqual(error.exception.code, 1)
                self.assertIn(diagnostic, stderr.getvalue())
                self.assertNotIn("secret-token", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())
                protect.assert_not_called()
                profile.assert_not_called()

            path.write_text(json.dumps(valid))
            project = root / "cvm_project.yml"
            project.write_text(
                json.dumps(
                    {
                        "trustee": {
                            "url": "https://user:secret-token@keys.test",
                            "ca": "ca.pem",
                            "admin_token_file": "admin.jwt",
                        },
                        "approval": {"public_keys": ["acceptance.pub"]},
                    }
                )
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                cli.main(["vault", str(path), "--project-config", str(project)])
            self.assertIn("HTTPS URL without credentials", stderr.getvalue())
            self.assertNotIn("secret-token", stderr.getvalue())

    def test_malformed_service_diagnostics_never_echo_service_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "vault_build.yml"
            service = root / "app_helper.service"
            (root / "application.tar").touch()
            path.write_text(
                json.dumps(
                    {
                        "image_id": "sha256:" + "a" * 64,
                        "docker_archive": "application.tar",
                        "cvm_image": str(root),
                        "services": [service.name],
                    }
                )
            )
            cases = {
                "missing section header": "secret-service-input\n[Service]\nExecStart=/vault/application/run\n",
                "malformed line": "[Service]\nExecStart=/vault/application/run\nsecret-service-input\n",
                "duplicate section": "[secret-service-input]\n[secret-service-input]\n",
                "duplicate option": "[Service]\nsecret-service-input=first\nsecret-service-input=second\n",
            }
            for case, text in cases.items():
                with self.subTest(case=case):
                    service.write_text(text)
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with (
                        patch.object(cli.vault.config, "project") as project,
                        patch.object(cli.vault, "profile_from_image") as profile,
                        patch.object(cli.vault, "protect_process") as protect,
                        patch.object(cli.vault, "memory_file") as key,
                        contextlib.redirect_stdout(stdout),
                        contextlib.redirect_stderr(stderr),
                        self.assertRaises(SystemExit) as error,
                    ):
                        cli.main(["vault", str(path)])
                    self.assertEqual(error.exception.code, 1)
                    self.assertIn("Invalid application service syntax", stderr.getvalue())
                    self.assertNotIn("secret-service-input", stdout.getvalue() + stderr.getvalue())
                    self.assertNotIn("Traceback", stdout.getvalue() + stderr.getvalue())
                    self.assertEqual(stdout.getvalue(), "")
                    for operation in (project, profile, protect, key):
                        operation.assert_not_called()

    def test_build_and_finalize_are_distinct_commands(self):
        with patch.object(cli.cvm, "build") as build, patch.object(cli, "report_bundle"):
            cli.main(["build", "profile.yml", "-p", "intel_tdx", "--defer-measurements", "--gpu", "0000:01:00.0"])
        build.assert_called_once_with(
            "profile.yml",
            "intel_tdx",
            None,
            defer_measurements=True,
            gpu=["0000:01:00.0"],
            dev=False,
            acceptance_runner=None,
            approval_key=None,
        )
        with patch.object(cli.cvm, "finalize") as finalize, patch.object(cli, "report_bundle"):
            cli.main(["finalize", "bundle", "--reference-evidence", "private.json"])
        finalize.assert_called_once_with("bundle", "private.json", None)

    def test_build_can_preflight_approve_enable_and_publish_in_one_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "cvm_profile" / "intel_tdx"
            bundle.mkdir(parents=True)
            (bundle / "approval.json").write_text("{}")
            (bundle / "cvm_manifest.json").write_text(
                json.dumps({"profile_version": "profile", "platform": "intel_tdx"})
            )
            admin_config = root / "admin.json"
            admin_config.write_text(json.dumps({"trustee": "config"}))
            profile = {
                "gpu": "none",
                "platforms": {"intel_tdx": {"firmware": Path("tdvf.fd")}},
            }
            with (
                patch.object(cli.config, "profile", return_value=profile),
                patch.object(cli.cvm, "select_platform", return_value="intel_tdx"),
                patch.object(cli, "check_host", return_value=True) as preflight,
                patch.object(cli.cvm, "build", return_value=bundle) as build,
                patch.object(cli, "report_bundle"),
                patch.object(cli.admin, "install") as install,
                patch.object(cli, "publish_generated") as publish,
            ):
                cli.main(
                    [
                        "build",
                        "profile.yml",
                        "-p",
                        "intel_tdx",
                        "--quote-probe",
                        "quote-probe",
                        "--acceptance-runner",
                        "acceptance-runner",
                        "--approval-key",
                        "approval.key",
                        "--install-config",
                        str(admin_config),
                        "--publish",
                        "registry.example/cvm/profile:release",
                        "--cosign-key",
                        "cosign.key",
                        "--publish-plain-http",
                    ]
                )
            preflight.assert_called_once_with(
                Path("tdvf.fd"), Path("quote-probe"), platform="intel_tdx", gpu=False, snp_vcek_store=None
            )
            build.assert_called_once_with(
                "profile.yml",
                "intel_tdx",
                None,
                defer_measurements=False,
                gpu=None,
                dev=False,
                acceptance_runner="acceptance-runner",
                approval_key="approval.key",
            )
            install.assert_called_once_with({"trustee": "config"}, bundle, candidate=False)
            publish.assert_called_once_with(
                bundle.parent,
                "registry.example/cvm/profile:release",
                True,
                "cosign.key",
                names=("cvm_profile_intel_tdx.oci.tar",),
            )

    def test_combined_publication_requires_signing_and_expands_each_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {
                "vault_deployment_amd_sev_snp.oci.tar": {},
                "vault_deployment_intel_tdx.oci.tar": {},
            }
            (root / "oci_artifacts.json").write_text(json.dumps({"schema_version": 1, "artifacts": artifacts}))
            with self.assertRaises(BuildError):
                cli.publish_generated(root, "registry.example/cvm/delivery:{artifact}", False, None)
            with self.assertRaises(BuildError):
                cli.publish_generated(root, "registry.example/cvm/delivery:release", False, "cosign.key")
            with patch.object(cli.oci, "publish", side_effect=("amd@sha256:a", "tdx@sha256:b")) as publish:
                references = cli.publish_generated(root, "registry.example/cvm/{artifact}:release", False, "cosign.key")
            self.assertEqual(references, ["amd@sha256:a", "tdx@sha256:b"])
            resolved = root.resolve()
            self.assertEqual(
                publish.call_args_list,
                [
                    call(
                        resolved / "vault_deployment_amd_sev_snp.oci.tar",
                        "registry.example/cvm/vault_deployment_amd_sev_snp:release",
                        False,
                        cosign_key="cosign.key",
                    ),
                    call(
                        resolved / "vault_deployment_intel_tdx.oci.tar",
                        "registry.example/cvm/vault_deployment_intel_tdx:release",
                        False,
                        cosign_key="cosign.key",
                    ),
                ],
            )
            with patch.object(cli.oci, "publish", return_value="tdx@sha256:b") as publish:
                cli.publish_generated(
                    root,
                    "registry.example/cvm/delivery:release",
                    False,
                    "cosign.key",
                    names=("vault_deployment_intel_tdx.oci.tar",),
                )
            publish.assert_called_once_with(
                resolved / "vault_deployment_intel_tdx.oci.tar",
                "registry.example/cvm/delivery:release",
                False,
                cosign_key="cosign.key",
            )

    def test_vault_dispatch_and_failure_do_not_disclose_secret_details(self):
        stderr = io.StringIO()
        with (
            patch.object(cli.vault, "build", side_effect=BuildError("sensitive-upload-token")) as build,
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as error,
        ):
            cli.main(["vault", "app.yml", "--project-config", "project.yml", "--output", "out", "--candidate"])
        build.assert_called_once_with("app.yml", "out", True, False, False, "project.yml")
        self.assertEqual(error.exception.code, 1)
        self.assertIn("build_failure.json", stderr.getvalue())
        self.assertNotIn("sensitive-upload-token", stderr.getvalue())

    def test_vault_can_publish_all_deliveries_in_the_same_command(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "oci_artifacts.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifacts": {"vault_deployment_intel_tdx.oci.tar": {"archive_sha256": "a" * 64}},
                    }
                )
            )
            with (
                patch.object(cli.vault, "build", return_value=output),
                patch.object(cli, "publish_generated") as publish,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                cli.main(
                    [
                        "vault",
                        "app.yml",
                        "--publish",
                        "registry.example/cvm/delivery:release",
                        "--cosign-key",
                        "cosign.key",
                        "--plain-http",
                    ]
                )
            publish.assert_called_once_with(output, "registry.example/cvm/delivery:release", True, "cosign.key")

    def test_corrupt_archive_diagnostic_reaches_cli_before_any_key_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "application.tar"
            archive.write_bytes(b"secret-corrupt-archive")
            app = {
                "cvm_image": "profile",
                "docker_archive": str(archive),
                "image_id": "sha256:" + "a" * 64,
                "requires_gpu": False,
                "allowed_out_ports": [443],
            }
            profiles = {"contract": {"gpu": "none", "bootstrap_egress": [443]}}
            stderr = io.StringIO()
            with (
                patch.object(cli.vault.config, "application", return_value=app),
                patch.object(
                    cli.vault.config, "project", return_value={"trustee": {}, "approval": {"public_keys": ["k"]}}
                ),
                patch.object(cli.vault.trustee_client, "admin_token"),
                patch.object(cli.vault, "profile_from_image", return_value=contextlib.nullcontext(profiles)),
                patch.object(cli.vault, "protect_process") as protect,
                patch.object(cli.vault, "memory_file") as key,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stderr),
                self.assertRaises(SystemExit) as error,
            ):
                cli.main(["vault", "app.yml"])
            self.assertEqual(error.exception.code, 1)
            self.assertIn("Invalid docker_archive; regenerate it with docker save", stderr.getvalue())
            self.assertNotIn("secret-corrupt-archive", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            protect.assert_not_called()
            key.assert_not_called()

    def test_host_preflight_without_quote_probe_retains_incomplete_status(self):
        with (
            patch.object(cli, "check_host", return_value=False) as check,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            cli.main(["preflight", "host", "--firmware", "tdvf.fd"])
        check.assert_called_once_with(Path("tdvf.fd"), None, platform="intel_tdx", gpu=False, snp_vcek_store=None)
        self.assertEqual(error.exception.code, 2)

    def test_approval_repackages_only_after_successful_validation(self):
        with (
            patch.object(cli, "read_json", return_value={}),
            patch.object(cli, "approve_bundle", side_effect=BuildError("rejected")) as approve,
            patch.object(cli, "package_bundle") as package,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            cli.main(["admin", "approve", "bundle", "evidence.json", "--signing-key", "acceptance.key"])
        self.assertEqual(error.exception.code, 1)
        approve.assert_called_once_with("bundle", {}, "acceptance.key")
        package.assert_not_called()
        # An acceptance signing key is mandatory; unsigned approval no longer exists.
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            cli.main(["admin", "approve", "bundle", "evidence.json"])
        self.assertEqual(error.exception.code, 2)

    def test_pull_requires_publisher_authentication_or_an_explicit_opt_out(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "delivery.oci.tar"
            archive.write_bytes(b"tar")
            registry = "registry.example.org/cvm/app@sha256:" + "a" * 64
            result = (Path(directory) / "out", {"artifactType": "t", "digest": "sha256:" + "b" * 64}, {})
            for argv, message in (
                (["pull", str(archive)], "--archive-sha256"),
                (["pull", registry], "--cosign-verification-key"),
            ):
                stderr = io.StringIO()
                with (
                    self.subTest(argv=argv),
                    patch.object(cli.oci, "materialize") as materialize,
                    contextlib.redirect_stderr(stderr),
                    self.assertRaises(SystemExit) as error,
                ):
                    cli.main(argv)
                self.assertEqual(error.exception.code, 1)
                self.assertIn(message, stderr.getvalue())
                materialize.assert_not_called()
            with (
                patch.object(cli.oci, "materialize", return_value=result) as materialize,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                cli.main(["pull", str(archive), "--archive-sha256", "c" * 64])
                self.assertEqual(materialize.call_args.kwargs, {"archive_sha256": "c" * 64, "cosign_key": None})
                cli.main(["pull", registry, "--cosign-verification-key", "release.pub", "--output", "out"])
                self.assertEqual(materialize.call_args.kwargs, {"archive_sha256": None, "cosign_key": "release.pub"})
                cli.main(["pull", str(archive), "--allow-unverified"])
                self.assertEqual(materialize.call_args.kwargs, {"archive_sha256": None, "cosign_key": None})

    def test_pull_prints_root_accessible_absolute_launch_command(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output with spaces"
            result = (
                output,
                {"artifactType": "t", "digest": "sha256:" + "b" * 64},
                {"launch_directory": "intel_tdx"},
            )
            stdout = io.StringIO()
            with patch.object(cli.oci, "materialize", return_value=result), contextlib.redirect_stdout(stdout):
                cli.main(["pull", "delivery.oci.tar", "--allow-unverified"])
            launch = str(output / "intel_tdx/launch_cvm.sh")
            self.assertIn(f"Launch: sudo -- '{launch}'", stdout.getvalue())
            self.assertNotIn("Launch: cd ", stdout.getvalue())

    def test_run_authenticates_materializes_and_launches_a_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "delivery"
            launcher = output / "intel_tdx" / "launch_cvm.sh"
            launcher.parent.mkdir(parents=True)
            launcher.touch()
            registry = "registry.example/cvm/app@sha256:" + "a" * 64
            result = (
                output,
                {"artifactType": cli.oci.DELIVERY_ARTIFACT_TYPE, "digest": "sha256:" + "b" * 64},
                {"launch_directory": "intel_tdx"},
            )
            with (
                patch.object(cli.oci, "materialize", return_value=result) as materialize,
                patch.object(cli.subprocess, "run") as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                cli.main(
                    [
                        "run",
                        registry,
                        "--cosign-verification-key",
                        "release.pub",
                        "--output",
                        str(output),
                        "--cvm-bundle",
                        "/opt/cvm/profile/intel_tdx",
                        "--gpu",
                        "0000:41:00.0",
                        "--bind-address",
                        "127.0.0.1",
                    ]
                )
            materialize.assert_called_once_with(
                registry,
                str(output),
                False,
                False,
                archive_sha256=None,
                cosign_key="release.pub",
            )
            run.assert_called_once_with(
                [
                    str(launcher.resolve()),
                    "--cvm-bundle",
                    "/opt/cvm/profile/intel_tdx",
                    "--gpu",
                    "0000:41:00.0",
                    "--bind-address",
                    "127.0.0.1",
                ],
                check=True,
            )

    def test_run_refuses_a_non_delivery_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            result = (
                Path(directory),
                {"artifactType": "bundle", "digest": "sha256:" + "b" * 64},
                {"kind": "cvm_bundle"},
            )
            with (
                patch.object(cli.oci, "materialize", return_value=result),
                patch.object(cli.subprocess, "run") as run,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                cli.main(
                    [
                        "run",
                        "registry.example/cvm/bundle@sha256:" + "a" * 64,
                        "--cosign-verification-key",
                        "release.pub",
                    ]
                )
            self.assertEqual(error.exception.code, 1)
            run.assert_not_called()

    def test_run_has_no_unverified_execution_path(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "delivery.oci.tar"
            archive.touch()
            with (
                patch.object(cli.oci, "materialize") as materialize,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                cli.main(["run", str(archive), "--allow-unverified"])
            self.assertEqual(error.exception.code, 2)
            materialize.assert_not_called()

            with (
                patch.object(cli.oci, "materialize") as materialize,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                cli.main(["run", str(archive)])
            self.assertEqual(error.exception.code, 1)
            materialize.assert_not_called()

    def test_run_rejects_an_unsafe_launch_directory_after_materialization(self):
        with tempfile.TemporaryDirectory() as directory:
            result = (
                Path(directory) / "delivery",
                {"artifactType": cli.oci.DELIVERY_ARTIFACT_TYPE, "digest": "sha256:" + "b" * 64},
                {"launch_directory": "../outside"},
            )
            with (
                patch.object(cli.oci, "materialize", return_value=result),
                patch.object(cli.subprocess, "run") as run,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                cli.main(
                    [
                        "run",
                        "registry.example/cvm/app@sha256:" + "a" * 64,
                        "--cosign-verification-key",
                        "release.pub",
                    ]
                )
            self.assertEqual(error.exception.code, 1)
            run.assert_not_called()

    def test_run_launcher_failure_is_reported_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "delivery"
            launcher = output / "intel_tdx" / "launch_cvm.sh"
            launcher.parent.mkdir(parents=True)
            launcher.touch()
            result = (
                output,
                {"artifactType": cli.oci.DELIVERY_ARTIFACT_TYPE, "digest": "sha256:" + "b" * 64},
                {"launch_directory": "intel_tdx"},
            )
            stderr = io.StringIO()
            with (
                patch.object(cli.oci, "materialize", return_value=result),
                patch.object(cli.subprocess, "run", side_effect=subprocess.CalledProcessError(1, [str(launcher)])),
                contextlib.redirect_stderr(stderr),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                cli.main(
                    [
                        "run",
                        "registry.example/cvm/app@sha256:" + "a" * 64,
                        "--cosign-verification-key",
                        "release.pub",
                    ]
                )
            self.assertEqual(error.exception.code, 1)
            self.assertIn("returned non-zero exit status", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_admin_acl_prints_a_bundle_scoped_resource_role(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.main(["admin", "acl", "cvm-0123abcd"])
        entry = json.loads(stdout.getvalue())
        self.assertEqual(entry["role"], "cvm-resources-cvm-0123abcd")
        self.assertIn("/keys/cvm\\-0123abcd/", entry["allowed_endpoints"])

    def test_diagnostics_directory_is_enabled_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(cli, "enable_diagnostics") as enable,
                patch.object(cli, "check_trustee") as check,
            ):
                cli.main(["--diagnostics", directory, "preflight", "trustee"])
            enable.assert_called_once_with(directory)
            check.assert_called_once()

    def test_references_dispatches_explicit_store_and_expiry(self):
        with patch.object(cli, "import_references") as publish:
            cli.main(
                ["references", "bundle", "--store", "store", "--state", "state", "--expires", "2026-12-01T00:00:00Z"]
            )
        publish.assert_called_once_with(Path("bundle"), Path("store"), Path("state"), "2026-12-01T00:00:00Z")

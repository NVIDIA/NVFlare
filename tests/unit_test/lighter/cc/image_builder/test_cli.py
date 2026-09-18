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
import unittest
from pathlib import Path
from unittest.mock import patch

from cvm import __main__ as cli
from cvm.common.errors import BuildError


class CliTests(unittest.TestCase):
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
        )
        with patch.object(cli.cvm, "finalize") as finalize, patch.object(cli, "report_bundle"):
            cli.main(["finalize", "bundle", "--reference-evidence", "private.json"])
        finalize.assert_called_once_with("bundle", "private.json", None)

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

    def test_host_preflight_without_quote_probe_retains_incomplete_status(self):
        with (
            patch.object(cli, "check_host", return_value=False) as check,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            cli.main(["preflight", "host", "--firmware", "tdvf.fd"])
        check.assert_called_once_with(Path("tdvf.fd"), None)
        self.assertEqual(error.exception.code, 2)

    def test_approval_repackages_only_after_successful_validation(self):
        with (
            patch.object(cli, "read_json", return_value={}),
            patch.object(cli, "approve_bundle", side_effect=BuildError("rejected")),
            patch.object(cli, "package_bundle") as package,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            cli.main(["admin", "approve", "bundle", "evidence.json"])
        self.assertEqual(error.exception.code, 1)
        package.assert_not_called()

    def test_references_dispatches_explicit_store_and_expiry(self):
        with patch.object(cli, "import_references") as publish:
            cli.main(
                ["references", "bundle", "--store", "store", "--state", "state", "--expires", "2026-12-01T00:00:00Z"]
            )
        publish.assert_called_once_with(Path("bundle"), Path("store"), Path("state"), "2026-12-01T00:00:00Z")

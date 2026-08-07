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

import subprocess
import textwrap
from unittest.mock import patch

import pytest

from nvflare.app_opt.confidential_computing.cc_authorizer import CCTokenGenerateError
from nvflare.app_opt.confidential_computing.tdx_authorizer import TDXAuthorizer

TOKEN = "eyJhbGciOiJFUzM4NCJ9.eyJpc3MiOiJ0cnVzdGF1dGhvcml0eSJ9.c2lnbmF0dXJl"
OTHER_TOKEN = "eyJhbGciOiJFUzM4NCJ9.eyJqdGkiOiJhbm90aGVyIn0.c2lnbmF0dXJl"


@pytest.fixture
def config_dir(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    return tmp_path


def test_generate_runs_unprivileged_and_returns_cli_output(config_dir):
    authorizer = TDXAuthorizer("/opt/tdx/bin/cli", str(config_dir), use_sudo=False)
    completed = subprocess.CompletedProcess([], 0, f"{TOKEN}\n", "")
    with patch("subprocess.run", return_value=completed) as run:
        assert authorizer.generate() == TOKEN

    command = run.call_args.args[0]
    assert command == [
        "/opt/tdx/bin/cli",
        "-c",
        str(config_dir / "config.json"),
        "token",
    ]
    assert run.call_args.kwargs["timeout"] == 60
    assert run.call_args.kwargs["start_new_session"] is True


def test_generate_can_explicitly_use_sudo(config_dir):
    authorizer = TDXAuthorizer("tdx-cli", str(config_dir), use_sudo=True)
    completed = subprocess.CompletedProcess([], 0, TOKEN, "")
    with patch("subprocess.run", return_value=completed) as run:
        authorizer.generate()
    assert run.call_args.args[0][:3] == ["sudo", "-n", "tdx-cli"]


def test_generate_auto_selects_noninteractive_sudo_for_non_root_host(config_dir):
    with patch("os.geteuid", return_value=1000), patch("shutil.which", return_value="/usr/bin/sudo"):
        authorizer = TDXAuthorizer("tdx-cli", str(config_dir))
    with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, TOKEN, "")) as run:
        authorizer.generate()
    assert run.call_args.args[0][:3] == ["sudo", "-n", "tdx-cli"]


def test_generate_supports_reviewed_version_specific_token_options(config_dir):
    authorizer = TDXAuthorizer("tdx-cli", str(config_dir), use_sudo=False, token_options=("--no-eventlog",))
    with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, TOKEN, "")) as run:
        assert authorizer.generate() == TOKEN
    assert run.call_args.args[0][-2:] == ["token", "--no-eventlog"]


def test_generate_extracts_one_jwt_without_accepting_other_stdout(config_dir):
    authorizer = TDXAuthorizer("tdx-cli", str(config_dir), use_sudo=False)
    completed = subprocess.CompletedProcess([], 0, f"informational line\n{TOKEN}\n", "")
    with patch("subprocess.run", return_value=completed):
        assert authorizer.generate() == TOKEN


def test_generate_rejects_ambiguous_stdout(config_dir):
    authorizer = TDXAuthorizer("tdx-cli", str(config_dir), use_sudo=False)
    completed = subprocess.CompletedProcess([], 0, f"{TOKEN}\n{OTHER_TOKEN}\n", "")
    with patch("subprocess.run", return_value=completed), pytest.raises(CCTokenGenerateError, match="exactly one"):
        authorizer.generate()


@pytest.mark.parametrize(
    "completed",
    [
        subprocess.CompletedProcess([], 1, "", "verification service unavailable"),
        subprocess.CompletedProcess([], 0, "", ""),
    ],
)
def test_generate_fails_closed(config_dir, completed):
    authorizer = TDXAuthorizer("tdx-cli", str(config_dir), use_sudo=False)
    with patch("subprocess.run", return_value=completed), pytest.raises(CCTokenGenerateError):
        authorizer.generate()


def test_verify_uses_exit_status(config_dir):
    authorizer = TDXAuthorizer("tdx-cli", str(config_dir), use_sudo=False)
    with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, "verified", "")) as run:
        assert authorizer.verify(TOKEN) is True
    assert run.call_args.args[0] == [
        "tdx-cli",
        "verify",
        "--config",
        str(config_dir / "config.json"),
        "--token",
        TOKEN,
    ]

    with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 2, "", "invalid token")):
        assert authorizer.verify(OTHER_TOKEN) is False


def test_verify_rejects_replayed_valid_token(config_dir):
    authorizer = TDXAuthorizer("tdx-cli", str(config_dir), use_sudo=False)
    verified = subprocess.CompletedProcess([], 0, "verified", "")
    with patch("subprocess.run", return_value=verified) as run:
        assert authorizer.verify(TOKEN) is True
        assert authorizer.verify(TOKEN) is False
    assert run.call_count == 2


def test_verify_retries_when_token_is_not_valid_yet(config_dir):
    authorizer = TDXAuthorizer("tdx-cli", str(config_dir), use_sudo=False)
    not_yet = subprocess.CompletedProcess([], 1, "", "token has invalid claims: token is not valid yet")
    verified = subprocess.CompletedProcess([], 0, "verified", "")

    with (
        patch("subprocess.run", side_effect=[not_yet, verified]) as run,
        patch("nvflare.app_opt.confidential_computing.tdx_authorizer.time.sleep") as sleep,
    ):
        assert authorizer.verify(TOKEN) is True

    assert run.call_count == 2
    sleep.assert_called_once_with(2.0)


def test_verify_exhausts_bounded_not_before_retry_window(config_dir):
    authorizer = TDXAuthorizer("tdx-cli", str(config_dir), use_sudo=False)
    not_yet = subprocess.CompletedProcess([], 1, "token has invalid claims: token is not valid yet", "")

    with (
        patch("subprocess.run", return_value=not_yet) as run,
        patch("nvflare.app_opt.confidential_computing.tdx_authorizer.time.sleep") as sleep,
    ):
        assert authorizer.verify(TOKEN) is False

    assert run.call_count == 5
    assert sleep.call_count == 4


def test_verify_does_not_retry_other_failures(config_dir):
    authorizer = TDXAuthorizer("tdx-cli", str(config_dir), use_sudo=False)
    invalid = subprocess.CompletedProcess([], 1, "", "signature verification failed")

    with (
        patch("subprocess.run", return_value=invalid) as run,
        patch("nvflare.app_opt.confidential_computing.tdx_authorizer.time.sleep") as sleep,
    ):
        assert authorizer.verify(TOKEN) is False

    run.assert_called_once()
    sleep.assert_not_called()


def test_verify_rejects_empty_token_without_starting_cli(config_dir):
    authorizer = TDXAuthorizer("tdx-cli", str(config_dir), use_sudo=False)
    with patch("subprocess.run") as run:
        assert authorizer.verify("") is False
    run.assert_not_called()


def test_timeout_fails_closed_for_generation_and_verification(config_dir):
    authorizer = TDXAuthorizer("tdx-cli", str(config_dir), cmd_timeout=0.01, use_sudo=False)
    timeout = subprocess.TimeoutExpired(["tdx-cli"], 0.01)

    with patch("subprocess.run", side_effect=timeout), pytest.raises(CCTokenGenerateError, match="timed out"):
        authorizer.generate()

    with patch("subprocess.run", side_effect=timeout):
        assert authorizer.verify(TOKEN) is False


def test_generate_and_verify_with_real_subprocess_contract(config_dir, tmp_path):
    cli = tmp_path / "tdx-cli"
    cli.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import sys

            if "token" in sys.argv:
                print("eyJhbGciOiJFUzM4NCJ9.eyJpc3MiOiJ0cnVzdGF1dGhvcml0eSJ9.c2lnbmF0dXJl")
                raise SystemExit(0)
            if "verify" in sys.argv and sys.argv[-1] == "eyJhbGciOiJFUzM4NCJ9.eyJpc3MiOiJ0cnVzdGF1dGhvcml0eSJ9.c2lnbmF0dXJl":
                raise SystemExit(0)
            print("invalid token", file=sys.stderr)
            raise SystemExit(2)
            """
        )
    )
    cli.chmod(0o755)
    authorizer = TDXAuthorizer(str(cli), str(config_dir), use_sudo=False)

    token = authorizer.generate()

    assert token == TOKEN
    assert authorizer.verify(token) is True
    assert authorizer.verify(OTHER_TOKEN) is False

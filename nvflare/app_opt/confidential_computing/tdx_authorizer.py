# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import hashlib
import logging
import os
import re
import shutil
import subprocess
import time

from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer, CCTokenGenerateError

from .utils import NonceHistory

TDX_NAMESPACE = "x-tdx"
TDX_CLI_CONFIG = "config.json"
TOKEN_NOT_VALID_YET = "token is not valid yet"
NOT_BEFORE_RETRY_INTERVAL = 2.0
NOT_BEFORE_VERIFY_ATTEMPTS = 5
MAX_TOKEN_SIZE = 256 * 1024
JWT_PATTERN = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")

# Retained for source compatibility. The implementation no longer writes
# shared result files, which made concurrent calls race with each other.
TOKEN_FILE = "token.txt"
VERIFY_FILE = "verify.txt"
ERROR_FILE = "error.txt"


class TDXAuthorizer(CCAuthorizer):
    """Generate and verify Intel TDX tokens with an operator-supplied CLI."""

    def __init__(
        self,
        tdx_cli_command: str,
        config_dir: str,
        cmd_timeout: float = 60,
        use_sudo: bool | None = None,
        max_nonce_history: int = 1000,
        max_token_size: int = MAX_TOKEN_SIZE,
        token_options: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Initialize the TDX authorizer.

        Args:
            tdx_cli_command: Path to the TDX attestation CLI executable.
            config_dir: Directory containing the CLI's ``config.json``.
            cmd_timeout: Maximum seconds allowed for each CLI invocation.
            use_sudo: Prefix the CLI with non-interactive ``sudo``. ``None``
                selects sudo for a non-root process when it is available, while
                root and minimal containers invoke the CLI directly.
            max_nonce_history: Number of successfully verified token
                fingerprints retained for replay rejection.
            max_token_size: Maximum accepted serialized JWT size in bytes.
            token_options: Options passed after ``token``. By default no TEE
                selector is supplied: the pinned Intel CLI selects TDX when no
                selector is present, and this remains compatible with older
                CLIs. Use this hook only for version-specific reviewed flags.
        """
        super().__init__()
        if not isinstance(tdx_cli_command, str) or not tdx_cli_command.strip():
            raise ValueError("tdx_cli_command must be a non-empty string")
        if cmd_timeout <= 0:
            raise ValueError("cmd_timeout must be positive")
        if max_token_size <= 0:
            raise ValueError("max_token_size must be positive")
        if use_sudo is not None and not isinstance(use_sudo, bool):
            raise ValueError("use_sudo must be true, false, or None")
        if token_options is None:
            token_options = ()
        if not isinstance(token_options, (list, tuple)) or any(
            not isinstance(option, str) or not option for option in token_options
        ):
            raise ValueError("token_options must contain only non-empty strings")

        self.tdx_cli_command = tdx_cli_command
        self.config_dir = os.path.abspath(config_dir)
        self.config_file = os.path.join(self.config_dir, TDX_CLI_CONFIG)
        self.cmd_timeout = cmd_timeout
        self.use_sudo = os.geteuid() != 0 and shutil.which("sudo") is not None if use_sudo is None else use_sudo
        self.max_token_size = max_token_size
        self.token_options = tuple(token_options)
        self.seen_token_history = NonceHistory(max_nonce_history)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _command(self, *arguments: str) -> list[str]:
        command = [self.tdx_cli_command, *arguments]
        return ["sudo", "-n", *command] if self.use_sudo else command

    def _extract_token(self, output: str) -> str:
        candidates = [line.strip() for line in output.splitlines() if JWT_PATTERN.fullmatch(line.strip())]
        if len(candidates) != 1:
            raise RuntimeError("TDX CLI output did not contain exactly one compact JWT")
        token = candidates[0]
        if len(token.encode("utf-8")) > self.max_token_size:
            raise RuntimeError("TDX CLI returned a token exceeding the configured size limit")
        return token

    def _run(self, command: list[str], action: str) -> subprocess.CompletedProcess:
        if not os.path.isfile(self.config_file):
            raise RuntimeError(f"TDX CLI configuration does not exist: {self.config_file}")
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.cmd_timeout,
                start_new_session=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"TDX CLI executable not found: {command[0]}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"TDX {action} timed out after {self.cmd_timeout} seconds") from e
        except OSError as e:
            raise RuntimeError(f"TDX {action} could not start: {e}") from e

    def generate(self) -> str:
        try:
            result = self._run(
                self._command("-c", self.config_file, "token", *self.token_options),
                "token generation",
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or f"exit status {result.returncode}"
                raise RuntimeError(detail)
            return self._extract_token(result.stdout)
        except RuntimeError as e:
            raise CCTokenGenerateError(f"Failed to generate a TDX token: {e}") from e

    def verify(self, token: str) -> bool:
        if (
            not isinstance(token, str)
            or not JWT_PATTERN.fullmatch(token.strip())
            or len(token.strip().encode("utf-8")) > self.max_token_size
        ):
            return False
        token = token.strip()
        command = self._command("verify", "--config", self.config_file, "--token", token)
        for attempt in range(NOT_BEFORE_VERIFY_ATTEMPTS):
            try:
                result = self._run(command, "token verification")
            except RuntimeError as e:
                self.logger.warning("TDX token verification failed: %s", e)
                return False

            if result.returncode == 0:
                token_fingerprint = hashlib.sha256(token.encode("utf-8")).hexdigest()
                if not self.seen_token_history.add(token_fingerprint):
                    self.logger.warning("Rejected a replayed TDX attestation token")
                    return False
                return True

            # Intel Trust Authority can issue a token whose not-before claim is
            # just ahead of the verifier clock. Retry only this known transient
            # result within a bounded window; every other verification failure
            # remains fail-closed.
            diagnostics = f"{result.stdout}\n{result.stderr}".lower()
            if attempt + 1 < NOT_BEFORE_VERIFY_ATTEMPTS and TOKEN_NOT_VALID_YET in diagnostics:
                self.logger.info(
                    "TDX token is not valid yet; retrying verification in %.1f seconds",
                    NOT_BEFORE_RETRY_INTERVAL,
                )
                time.sleep(NOT_BEFORE_RETRY_INTERVAL)
                continue

            # A verifier can echo its token in diagnostics. Do not copy stderr
            # into NVFlare logs, where an attestation bearer token could leak.
            self.logger.warning("TDX token verification failed with exit status %d", result.returncode)
            return False
        return False

    def get_namespace(self) -> str:
        return TDX_NAMESPACE

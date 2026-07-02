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

import os
import subprocess
import tempfile
from typing import Mapping, Sequence
from urllib.parse import urlparse

from nvflare.fuel.sec.admin_cert import ADMIN_CERT_PLACEHOLDER_CN
from nvflare.fuel.sec.ephemeral_admin_cert import EphemeralAdminCertError, EphemeralAdminCertFiles

DEFAULT_STEP_CA_CERT_TTL = "24h"
DEFAULT_STEP_CA_REQUEST_NAME = ADMIN_CERT_PLACEHOLDER_CN
DEFAULT_STEP_CA_COMMAND_TIMEOUT = 300.0


def obtain_step_ca_admin_cert_files(config: Mapping, root_ca_file: str) -> EphemeralAdminCertFiles:
    temp_dir = tempfile.TemporaryDirectory(prefix="nvflare-step-ca-admin-")
    cert_path = os.path.join(temp_dir.name, "client.crt")
    key_path = os.path.join(temp_dir.name, "client.key")
    command = _build_step_ca_command(
        config=config,
        root_ca_file=root_ca_file,
        cert_path=cert_path,
        key_path=key_path,
    )
    command_timeout = float(config.get("command_timeout") or DEFAULT_STEP_CA_COMMAND_TIMEOUT)

    try:
        _run_step(command, timeout=command_timeout)
    except Exception:
        temp_dir.cleanup()
        raise

    return EphemeralAdminCertFiles(
        client_key=key_path,
        client_cert=cert_path,
        temp_dir=temp_dir,
    )


def validate_step_ca_admin_cert_config(config: Mapping) -> dict:
    if not isinstance(config, Mapping):
        raise EphemeralAdminCertError(f"step_ca provider_config must be a mapping but got {type(config)}")
    result = dict(config)
    ca_url = result.get("ca_url")
    if not ca_url:
        raise EphemeralAdminCertError("step_ca provider_config.ca_url is required")
    _validate_step_ca_url(str(ca_url))
    if not result.get("provisioner"):
        raise EphemeralAdminCertError("step_ca provider_config.provisioner is required")
    _command_timeout(result)
    return result


def _build_step_ca_command(
    config: Mapping,
    root_ca_file: str,
    cert_path: str,
    key_path: str,
) -> Sequence[str]:
    config = validate_step_ca_admin_cert_config(config)
    step_bin = str(config.get("step_bin") or "step")
    ca_url = str(config.get("ca_url"))
    provisioner = str(config.get("provisioner"))
    command = [step_bin, "ca", "certificate", "--ca-url", ca_url, "--root", root_ca_file]
    command.extend(["--provisioner", provisioner])
    command.extend(["--not-after", str(config.get("cert_ttl") or DEFAULT_STEP_CA_CERT_TTL)])
    command.extend(["--kty", "RSA", "--size", "2048"])
    command.extend([DEFAULT_STEP_CA_REQUEST_NAME, cert_path, key_path])
    return command


def _command_timeout(config: Mapping) -> float:
    command_timeout_config = config.get("command_timeout")
    if command_timeout_config is None or command_timeout_config == "":
        return DEFAULT_STEP_CA_COMMAND_TIMEOUT
    try:
        command_timeout = float(command_timeout_config)
    except (TypeError, ValueError) as ex:
        raise EphemeralAdminCertError("step_ca provider_config.command_timeout must be a number") from ex
    if command_timeout <= 0.0:
        raise EphemeralAdminCertError("step_ca provider_config.command_timeout must be greater than zero")
    return command_timeout


def _run_step(
    command: Sequence[str],
    timeout: float = DEFAULT_STEP_CA_COMMAND_TIMEOUT,
):
    try:
        return subprocess.run(
            command,
            check=True,
            timeout=timeout,
        )
    except FileNotFoundError as ex:
        raise EphemeralAdminCertError(
            "step-ca admin certs require the 'step' CLI in PATH or step_ca provider_config.step_bin"
        ) from ex
    except subprocess.TimeoutExpired as ex:
        raise EphemeralAdminCertError(f"step ca certificate timed out after {timeout} seconds") from ex
    except subprocess.CalledProcessError as ex:
        raise EphemeralAdminCertError(f"step ca certificate failed with exit code {ex.returncode}") from ex


def _validate_step_ca_url(url: str):
    parsed = urlparse(url)
    if parsed.scheme == "https" and parsed.netloc:
        return
    if parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1", "localhost"}:
        return
    raise EphemeralAdminCertError("step_ca provider_config.ca_url must use https; http is only allowed for localhost")

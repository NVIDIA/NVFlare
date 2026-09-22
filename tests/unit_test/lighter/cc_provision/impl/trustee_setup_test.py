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

"""Execute pinned Trustee setup with disposable keys, never Docker or real credentials."""

import hashlib
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[5] / "examples/devops/coco/service"
HARDENER = SERVICE / "lib/harden-trustee-setup.py"
# Exact upstream setup.sh at confidential-containers/trustee commit
# 338610fbfed57b66c61a8a3a60e0e4386bdce793; the production helper pins its hash.
UPSTREAM = r"""#!/bin/ash
set -euxo pipefail

KEY_DIR="/opt/confidential-containers/kbs/user-keys"
cd "${KEY_DIR}"

# Token-signing cert validity in days; defaults to 90 instead of openssl's silent 30.
TOKEN_CERT_DAYS="${TOKEN_CERT_DAYS:-90}"

# Root CA cert validity in days; default unchanged at 10 years.
CA_CERT_DAYS="${CA_CERT_DAYS:-3650}"

if [ ! -s private.key ]; then
  openssl genpkey -algorithm ed25519 > private.key
  openssl pkey -in private.key -pubout -out public.pub
fi

if [ ! -s admin-token ]; then
  b64url() {
    openssl base64 -A | tr '+/' '-_' | tr -d '='
  }

  header='{"alg":"EdDSA","typ":"JWT"}'
  iat="$(date +%s)"
  exp="$((iat + 315360000))" # 10 years
  payload="{\"iss\":\"TrusteeInDocker\",\"sub\":\"admin\",\"aud\":[\"KBS\"],\"role\":\"admin\",\"iat\":${iat},\"exp\":${exp}}"

  h64="$(printf '%s' "${header}" | b64url)"
  p64="$(printf '%s' "${payload}" | b64url)"
  signing_input="$(mktemp)"
  printf '%s' "${h64}.${p64}" > "${signing_input}"
  sig="$(openssl pkeyutl -sign -inkey private.key -rawin -in "${signing_input}" | b64url)"
  rm -f "${signing_input}"
  printf '%s.%s.%s\n' "${h64}" "${p64}" "${sig}" > admin-token
  chmod 644 admin-token
  echo "Generated ${KEY_DIR}/admin-token (10 years validity)."
fi

if [ ! -s token.key ]; then
  openssl genrsa -traditional -out ca.key 2048
  openssl req -new -key ca.key -out ca-req.csr -subj "/O=CNCF/OU=CoCo/CN=KBS-compose-root"
  openssl req -x509 -days "${CA_CERT_DAYS}" -key ca.key -in ca-req.csr -out ca-cert.pem
  openssl ecparam -name prime256v1 -genkey -noout -out token.key
  openssl req -new -key token.key -out token-req.csr -subj "/O=CNCF/OU=CoCo/CN=CoCo-AS"
  openssl x509 -req -days "${TOKEN_CERT_DAYS}" -in token-req.csr -CA ca-cert.pem -CAkey ca.key -CAcreateserial -out token-cert.pem -extensions req_ext
  cat token-cert.pem ca-cert.pem > token-cert-chain.pem
fi
"""


def harden(path):
    return subprocess.run([sys.executable, str(HARDENER), str(path)], capture_output=True, text=True, timeout=10)


def setup(tmp_path, *, patched=True):
    key_dir = tmp_path / "keys"
    key_dir.mkdir(mode=0o755)
    path = key_dir / "setup.sh"
    path.write_text(UPSTREAM)
    if patched:
        result = harden(path)
        assert result.returncode == 0, result.stderr
    # Only the test's mount location changes; execute the actual patched source.
    path.write_text(path.read_text().replace("/opt/confidential-containers/kbs/user-keys", str(key_dir)))
    return path


def run_setup(path, *, shell=None, extra_env=None):
    shell = shell or ["bash"]
    if not shutil.which(shell[0]) or not shutil.which("openssl"):
        pytest.skip("Trustee setup regression requires a local shell and OpenSSL")
    command = "umask 022; exec " + shlex.join([*shell, str(path)])
    return subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        env={**os.environ, **(extra_env or {})},
        timeout=30,
    )


def test_fixture_is_exact_pinned_upstream():
    assert hashlib.sha256(UPSTREAM.encode()).hexdigest() == (
        "c299639734bf68c7532a43aa403f77eacd13af976fd94074cab0b58c4e3a5768"
    )


def test_upstream_reproduces_token_disclosure_and_readable_mode(tmp_path):
    path = setup(tmp_path, patched=False)
    result = run_setup(path)
    assert result.returncode == 0
    token = (path.parent / "admin-token").read_text().strip()
    # printf traces the three arguments separately; together they reconstruct
    # the complete usable JWT even if no single log line contains its dots.
    assert all(fragment in result.stderr for fragment in token.split("."))
    assert (path.parent / "admin-token").stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize("shell", [["bash"], ["bash", "-x"], ["busybox", "ash", "-x"]])
def test_hardened_setup_keeps_secrets_out_of_logs_and_private_from_creation(tmp_path, shell):
    path = setup(tmp_path)
    wrappers = tmp_path / "bin"
    wrappers.mkdir()
    # Observe permissions while redirections already exist, BEFORE the script's
    # final chmod. Real OpenSSL still creates and signs all disposable material.
    for program in ("openssl", "chmod"):
        real = shutil.which(program)
        if not real:
            pytest.skip(f"Trustee setup regression requires {program}")
        probe = wrappers / program
        probe.write_text(
            f"#!{sys.executable}\n"
            "import os, sys\n"
            f"program = {program!r}\n"
            "if program == 'openssl' and sys.argv[1:2] == ['genpkey']:\n"
            "    mode = os.fstat(1).st_mode & 0o777\n"
            "    with open(os.environ['MODE_LOG'], 'a') as f: f.write(f'private.key:{mode:o}\\n')\n"
            "if program == 'chmod' and sys.argv[1:] == ['0600', 'admin-token']:\n"
            "    mode = os.stat('admin-token').st_mode & 0o777\n"
            "    with open(os.environ['MODE_LOG'], 'a') as f: f.write(f'admin-token:{mode:o}\\n')\n"
            f"os.execv({real!r}, [{real!r}, *sys.argv[1:]])\n"
        )
        probe.chmod(0o700)
    log = tmp_path / "modes"
    result = run_setup(path, shell=shell, extra_env={"PATH": f"{wrappers}:{os.environ['PATH']}", "MODE_LOG": str(log)})
    assert result.returncode == 0, result.stderr
    observations = log.read_text().splitlines()
    assert observations[0] == "private.key:600"
    # BusyBox ash can dispatch chmod as a built-in applet, bypassing PATH.
    if shell[0] == "bash":
        assert observations == ["private.key:600", "admin-token:600"]
    token = (path.parent / "admin-token").read_text().strip()
    # Even reconstructible JWT fragments must not appear in Docker-captured output.
    for fragment in [token, *token.split(".")]:
        assert fragment not in result.stdout + result.stderr
    assert "signing_input=" not in result.stderr
    for secret in ("private.key", "admin-token", "ca.key", "token.key"):
        assert (path.parent / secret).stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    # Existing secrets are not rotated silently; inherited tracing stays off and
    # permissions are repaired for deployments created by the old setup script.
    original_key = (path.parent / "private.key").read_bytes()
    (path.parent / "admin-token").chmod(0o644)
    rerun = run_setup(path, shell=shell)
    assert rerun.returncode == 0, rerun.stderr
    assert (path.parent / "private.key").read_bytes() == original_key
    assert (path.parent / "admin-token").read_text().strip() == token
    assert (path.parent / "admin-token").stat().st_mode & 0o777 == 0o600
    assert token not in rerun.stdout + rerun.stderr


def test_patch_is_idempotent_and_stage_runs_it_before_compose(tmp_path):
    path = tmp_path / "setup.sh"
    path.write_text(UPSTREAM)
    assert harden(path).returncode == 0
    content = path.read_bytes()
    result = harden(path)
    assert result.returncode == 0 and "already hardened" in result.stdout
    assert path.read_bytes() == content
    stage = (SERVICE / "05-deploy-trustee.sh").read_text()
    assert stage.index('python3 "${SCRIPT_DIR}/lib/harden-trustee-setup.py"') < stage.index("sudo docker compose")


@pytest.mark.parametrize("patched", [False, True])
def test_unreviewed_source_is_rejected_without_mutation(tmp_path, patched):
    path = tmp_path / "setup.sh"
    path.write_text(UPSTREAM)
    if patched:
        assert harden(path).returncode == 0
    path.write_text(path.read_text() + "\necho unexpected\n")
    before = path.read_bytes()
    result = harden(path)
    assert result.returncode != 0 and "unreviewed source" in result.stderr
    assert path.read_bytes() == before


@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_linked_setup_is_rejected(tmp_path, link):
    source = tmp_path / "original"
    source.write_text(UPSTREAM)
    path = tmp_path / "setup.sh"
    if link == "symlink":
        path.symlink_to(source)
    else:
        os.link(source, path)
    result = harden(path)
    assert result.returncode != 0 and "non-linked file" in result.stderr
    assert source.read_text() == UPSTREAM


def test_hardening_does_not_hide_setup_failure(tmp_path):
    path = setup(tmp_path)
    wrappers = tmp_path / "bin"
    wrappers.mkdir()
    failing = wrappers / "openssl"
    failing.write_text("#!/bin/sh\necho 'fixture signing failure' >&2\nexit 73\n")
    failing.chmod(0o700)
    result = run_setup(path, extra_env={"PATH": f"{wrappers}:{os.environ['PATH']}"})
    assert result.returncode == 73
    assert "fixture signing failure" in result.stderr
    assert not (path.parent / "admin-token").exists()

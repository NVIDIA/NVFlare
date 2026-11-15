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

"""One Trustee proof format for CoCo guests and bare-metal CVMs."""

import json
import math
import os
import stat
import subprocess
import tempfile
import time

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from .cc_authorizer import CCTokenGenerateError
from .coco_authorizer import MAX_TOKEN_BYTES, CoCoAuthorizer, _TemporaryTokenError


class TrusteeAuthorizer(CoCoAuthorizer):
    """Generate the same identity-bound Trustee proof through either guest integration.

    CoCo obtains an EAR and its TEE key from the guest Attestation Agent. A
    bare-metal CVM invokes the measured kbs-client directly with an ephemeral
    key. Verifier-only instances need neither provider.
    """

    def __init__(
        self,
        *args,
        token_provider="verifier",
        kbs_url=None,
        kbs_ca=None,
        kbs_client="/host/bin/kbs-client",
        guest_loader="/host/lib/ld-linux-x86-64.so.2",
        guest_token_file=None,
        **kwargs,
    ):
        if token_provider not in ("verifier", "coco", "cvm"):
            raise ValueError("token_provider must be verifier, coco, or cvm")
        if token_provider == "cvm":
            if guest_token_file is not None:
                if not isinstance(guest_token_file, str) or not guest_token_file.startswith("/"):
                    raise ValueError("guest_token_file must be an absolute guest path")
            else:
                if not isinstance(kbs_url, str) or not kbs_url.startswith("https://"):
                    raise ValueError("CVM Trustee token generation requires an HTTPS kbs_url")
                if not isinstance(kbs_ca, str) or "BEGIN CERTIFICATE" not in kbs_ca:
                    raise ValueError("CVM Trustee token generation requires a PEM kbs_ca")
                if not isinstance(kbs_client, str) or not kbs_client.startswith("/"):
                    raise ValueError("kbs_client must be an absolute guest path")
                if not isinstance(guest_loader, str) or not guest_loader.startswith("/"):
                    raise ValueError("guest_loader must be an absolute guest path")
        self.token_provider = token_provider
        self.kbs_url = kbs_url
        self.kbs_ca = kbs_ca
        self.kbs_client = kbs_client
        self.guest_loader = guest_loader
        self.guest_token_file = guest_token_file
        super().__init__(*args, **kwargs)

    def _get_guest_token(self):
        if self.token_provider == "coco":
            return super()._get_guest_token()
        if self.token_provider != "cvm":
            raise CCTokenGenerateError("Verifier-only Trustee authorizer cannot generate tokens")

        if self.guest_token_file:
            try:
                descriptor = os.open(self.guest_token_file, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    metadata = os.fstat(descriptor)
                    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_TOKEN_BYTES + 16384:
                        raise ValueError("Invalid guest token file")
                    with os.fdopen(descriptor) as stream:
                        descriptor = -1
                        reply = json.load(stream)
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
                if not isinstance(reply, dict) or set(reply) != {"token", "tee_keypair"}:
                    raise ValueError("Invalid guest token file")
                return reply
            except OSError:
                raise _TemporaryTokenError("CVM guest token temporarily unavailable") from None

        deadline = getattr(self._generation_context, "deadline", float("inf"))
        remaining = deadline - time.monotonic() if math.isfinite(deadline) else 60.0
        if remaining <= 0:
            raise CCTokenGenerateError("Token generation deadline exhausted")
        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        key_pem = private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
        try:
            with tempfile.TemporaryDirectory(prefix="nvflare-trustee-") as directory, tempfile.TemporaryFile() as key:
                ca_path = os.path.join(directory, "kbs-ca.pem")
                key.write(key_pem)
                key.flush()
                key.seek(0)
                fd = os.open(ca_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(self.kbs_ca.encode())
                remaining = deadline - time.monotonic() if math.isfinite(deadline) else 60.0
                if remaining <= 0:
                    raise CCTokenGenerateError("Token generation deadline exhausted")
                environment = {name: value for name, value in os.environ.items() if not name.startswith("LD_")}
                environment.update(RUST_LOG="off", LD_LIBRARY_PATH="/host/lib")
                result = subprocess.run(
                    [
                        self.guest_loader,
                        "--library-path",
                        "/host/lib",
                        self.kbs_client,
                        "--url",
                        self.kbs_url,
                        "--cert-file",
                        ca_path,
                        "attest",
                        "--tee-key-file",
                        f"/proc/self/fd/{key.fileno()}",
                    ],
                    pass_fds=(key.fileno(),),
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=max(0.1, remaining),
                    env=environment,
                )
            if result.returncode != 0 or not result.stdout.strip():
                raise _TemporaryTokenError("CVM KBS attestation temporarily unavailable")
            return {"token": result.stdout.strip(), "tee_keypair": key_pem.decode()}
        except (OSError, subprocess.TimeoutExpired):
            raise _TemporaryTokenError("CVM KBS attestation temporarily unavailable") from None

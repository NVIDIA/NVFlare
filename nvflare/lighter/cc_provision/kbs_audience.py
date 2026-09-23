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

"""Read-only live KBS admin-audience probe; Python stdlib and OpenSSL only."""

import base64
import json
import os
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def b64(value):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def admin_token(private_key, audience):
    now = int(time.time())
    claims = {"iss": "TrusteeInDocker", "sub": "admin", "role": "admin", "iat": now, "exp": now + 60}
    if audience is not None:
        claims["aud"] = [audience]
    unsigned = b64(b'{"alg":"EdDSA","typ":"JWT"}') + "." + b64(json.dumps(claims).encode())
    with tempfile.TemporaryDirectory(prefix="kbs-audience-") as work:
        data = Path(work) / "input"
        data.write_bytes(unsigned.encode())
        data.chmod(0o600)
        signature = subprocess.run(
            ["openssl", "pkeyutl", "-sign", "-inkey", str(private_key), "-rawin", "-in", str(data)],
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
    return unsigned + "." + b64(signature)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward the admin credential to another endpoint.


def probe(url, ca_file, private_key):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
        raise ValueError("a credential-free HTTPS KBS URL is required")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        NoRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=str(ca_file))),
    )
    for label, audience, expected in (
        ("valid audience", "KBS", 200),
        ("wrong audience", "not-KBS", 401),
        ("missing audience", None, 401),
        ("no credential", None, 401),
    ):
        headers = {} if label == "no credential" else {"Authorization": "Bearer " + admin_token(private_key, audience)}
        request = urllib.request.Request(url.rstrip("/") + "/kbs/v0/resource-policy", headers=headers)
        try:
            with opener.open(request, timeout=15) as response:
                status = response.status
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
        if status != expected:
            raise RuntimeError(f"KBS {label}: expected HTTP {expected}, received {status}")
        print(f"KBS {label}: HTTP {status}")


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: kbs-admin-audience.py HTTPS-URL CA-FILE ADMIN-PRIVATE-KEY")
    os.umask(0o077)
    probe(*sys.argv[1:])


if __name__ == "__main__":
    main()

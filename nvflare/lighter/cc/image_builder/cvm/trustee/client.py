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

"""Authenticated native CoCo Trustee administration client."""

import base64
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from ..common.contracts import validate_resource
from ..common.errors import BuildError, require
from ..common.io import canonical


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise BuildError("Administrative redirects are forbidden")


def encode(data):
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def api(config, method, endpoint, data=None, *, content_type="application/json"):
    require(config["url"].startswith("https://"), "KBS administration requires HTTPS")
    if "admin_token_file" in config:
        require("admin_private_key" not in config, "Select one Trustee administration credential")
        token = Path(config["admin_token_file"]).read_text().strip()
        require(
            token and len(token) <= 16384 and token.isascii() and not any(c.isspace() for c in token),
            "Invalid Trustee administration token",
        )
    else:
        key = serialization.load_pem_private_key(Path(config["admin_private_key"]).read_bytes(), password=None)
        require(isinstance(key, ed25519.Ed25519PrivateKey), "KBS administration requires an Ed25519 key")
        now = int(time.time())
        body = (
            encode(canonical({"alg": "EdDSA", "typ": "JWT"}))
            + "."
            + encode(
                canonical(
                    {
                        "iat": now,
                        "nbf": now - 5,
                        "exp": now + 60,
                        "role": config.get("admin_role", "cvm-policy"),
                        "iss": config.get("admin_issuer", "cvm-builder"),
                        "aud": config.get("admin_audience", "coco-trustee"),
                    }
                )
            )
        )
        token = body + "." + encode(key.sign(body.encode()))
    context = ssl.create_default_context(cafile=config["ca"])
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context), NoRedirect())
    request = urllib.request.Request(
        config["url"].rstrip("/") + "/kbs/v0/" + endpoint,
        data=data,
        method=method,
        headers={"Authorization": "Bearer " + token, "Content-Type": content_type},
    )
    try:
        with opener.open(request, timeout=30) as response:
            result = response.read(8 * 1024**2 + 1)
            require(len(result) <= 8 * 1024**2, "Administrative response too large")
            return result
    except urllib.error.HTTPError as exc:
        exc.close()
        raise BuildError("KBS administrative request rejected; no credentials logged") from None
    except (urllib.error.URLError, OSError):
        raise BuildError("KBS administrative request failed; no credentials logged") from None


def upload_resource(config, resource, secret):
    """Upload one new vault key; native Trustee POST permits replacement.

    Call only for a freshly sealed vault identity. Never replay an abandoned
    upload after revocation. There is no client-side check-and-set emulation.
    """
    validate_resource(resource)
    require(len(secret) == 64, "Vault secret must be exactly 64 bytes")
    api(config, "POST", "resource/" + resource, secret, content_type="application/octet-stream")


def delete_resource(config, resource):
    """Delete a native resource; pending uploads must be fenced by the operator."""
    validate_resource(resource)
    api(config, "DELETE", "resource/" + resource)

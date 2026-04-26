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

"""Certificate fingerprint helpers for distributed provisioning."""

import re

from cryptography.hazmat.primitives import hashes

_SHA256_HEX_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def _format_sha256_fingerprint(digest_hex: str) -> str:
    digest = digest_hex.upper()
    return "SHA256:" + ":".join(digest[i : i + 2] for i in range(0, len(digest), 2))


def cert_fingerprint_sha256(cert) -> str:
    """Return an OpenSSL-style SHA256 fingerprint for a loaded x509 certificate."""
    return _format_sha256_fingerprint(cert.fingerprint(hashes.SHA256()).hex())


def normalize_sha256_fingerprint(value: str) -> str:
    """Normalize common SHA256 certificate fingerprint forms to ``SHA256:AA:BB...``.

    Accepted forms include:
    - ``sha256 Fingerprint=AA:BB:...`` from ``openssl x509 -fingerprint -sha256``
    - ``SHA256:AA:BB:...``
    - ``SHA256=AA:BB:...``
    - plain 64-character hex strings

    Returns an empty string when the value is not a valid SHA256 fingerprint.
    """
    if not isinstance(value, str):
        return ""
    text = value.strip()
    text = re.sub(r"^\s*sha256\s+fingerprint\s*=\s*", "", text, count=1, flags=re.IGNORECASE)
    text = re.sub(r"^\s*sha256\s*[:=]\s*", "", text, count=1, flags=re.IGNORECASE)
    compact = re.sub(r"[^0-9a-fA-F]", "", text)
    if not _SHA256_HEX_PATTERN.fullmatch(compact):
        return ""
    return _format_sha256_fingerprint(compact)

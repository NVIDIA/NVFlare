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

"""Create native Trustee operator inputs without a CoCo deployment."""

import os
import re
import tempfile
import time
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from ..common.errors import require
from ..common.io import canonical, digest_file, is_sha256, read_json, write_json
from ..common.versions import (
    COCO_ACTIX_HTTP_CRATE_SHA256,
    COCO_ACTIX_HTTP_SOURCE_SHA256,
    COCO_ACTIX_HTTP_VERSION,
    COCO_SERVICE_BUILD_PROFILE,
    COCO_SERVICE_PATCH_SHA256,
)
from .client import MAX_ADMIN_TOKEN_LIFETIME_SECONDS, encode

RECEIPT_PROVENANCE_FIELDS = (
    "trustee_commit",
    "source_clean",
    "build_profile",
    "source_patch_sha256",
    "actix_http_version",
    "actix_http_crate_sha256",
    "actix_http_source_sha256",
    "kbs_image_id",
)


def _validate_build_provenance(build):
    require(isinstance(build.get("trustee_commit"), str), "Invalid Trustee build provenance")
    if build.get("source_clean") is True:
        return
    expected = {
        "build_profile": COCO_SERVICE_BUILD_PROFILE,
        "source_patch_sha256": COCO_SERVICE_PATCH_SHA256,
        "actix_http_version": COCO_ACTIX_HTTP_VERSION,
        "actix_http_crate_sha256": COCO_ACTIX_HTTP_CRATE_SHA256,
        "actix_http_source_sha256": COCO_ACTIX_HTTP_SOURCE_SHA256,
    }
    image = build.get("kbs_image_id", "")
    require(
        build.get("source_clean") is False
        and all(build.get(name) == value for name, value in expected.items())
        and isinstance(image, str)
        and image.startswith("sha256:")
        and is_sha256(image.removeprefix("sha256:")),
        "Trustee build provenance is not an approved clean or CoCo service build",
    )


def _write_private(path, data):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".cvm-token-", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def issue_admin_token(
    private_key, role, output, *, issuer="cvm-builder", audience="coco-trustee", lifetime=3600, now=None
):
    """Issue one short-lived EdDSA token for an explicitly selected KBS role."""
    require(
        role in ("cvm-policy", "cvm-resources") or re.fullmatch(r"cvm-resources-[a-z0-9][a-z0-9_-]{0,63}", role),
        "Unsupported Trustee administration role",
    )
    require(
        type(lifetime) is int and 0 < lifetime <= MAX_ADMIN_TOKEN_LIFETIME_SECONDS,
        "Trustee administration token lifetime must be from 1 second through 30 days",
    )
    require(isinstance(issuer, str) and issuer and isinstance(audience, str) and audience, "Invalid token identity")
    key = serialization.load_pem_private_key(Path(private_key).read_bytes(), password=None)
    require(isinstance(key, ed25519.Ed25519PrivateKey), "KBS administration requires an Ed25519 key")
    issued = int(time.time() if now is None else now)
    header = encode(canonical({"alg": "EdDSA", "typ": "JWT"}))
    claims = encode(
        canonical(
            {
                "iat": issued,
                "nbf": issued - 5,
                "exp": issued + lifetime,
                "role": role,
                "iss": issuer,
                "aud": audience,
            }
        )
    )
    body = header + "." + claims
    token = body + "." + encode(key.sign(body.encode()))
    return _write_private(output, (token + "\n").encode())


def _policy_mapping(values):
    policies = {}
    for item in values:
        name, separator, path = item.partition("=")
        require(separator and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", name), "Use --policy NAME=FILE")
        require(name not in policies, "Duplicate immutable AS policy: " + name)
        source = Path(path).expanduser().resolve()
        require(source.is_file(), "Immutable AS policy is missing: " + str(source))
        policies[name] = digest_file(source)
    require(policies, "Supply every installed CVM AS policy with --policy NAME=FILE")
    return policies


def create_deployment_receipt(
    trustee_build,
    kbs_config,
    policies,
    output,
    *,
    policy_selection_tested=False,
    unauthorized_administration_denied=False,
):
    """Record the exact native deployment after its administrative tests pass."""
    require(policy_selection_tested, "Run policy-selection acceptance, then pass --policy-selection-tested")
    require(
        unauthorized_administration_denied,
        "Run the unauthorized-administration negative test, then pass --unauthorized-administration-denied",
    )
    build = read_json(trustee_build)
    _validate_build_provenance(build)
    config = read_json(kbs_config)
    service = config.get("attestation_service", {})
    policy_id_map = service.get("policy_id_map")
    require(isinstance(policy_id_map, dict) and policy_id_map, "KBS configuration has no policy_id_map")
    require(
        all(
            isinstance(selector, str)
            and selector
            and isinstance(ids, list)
            and ids
            and all(isinstance(policy_id, str) and policy_id for policy_id in ids)
            for selector, ids in policy_id_map.items()
        ),
        "KBS policy_id_map is invalid",
    )
    receipt = {name: build[name] for name in RECEIPT_PROVENANCE_FIELDS if name in build}
    receipt.update(
        {
            "policy_selection_tested": True,
            "unauthorized_administration_denied": True,
            "policy_id_map": policy_id_map,
            "immutable_as_policies": _policy_mapping(policies),
        }
    )
    write_json(output, receipt, mode=0o600)
    return Path(output).expanduser().resolve()

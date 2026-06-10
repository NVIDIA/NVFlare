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

import hashlib
import time
from typing import Mapping, Optional

from nvflare.apis.job_def import ALL_SITES, JobMetaKey
from nvflare.fuel.f3.cellnet.identity import get_cert_common_name
from nvflare.fuel.sec.principal import to_tuple
from nvflare.fuel.utils.json_utils import canonical_json
from nvflare.lighter.utils import (
    cert_not_valid_after_utc,
    cert_not_valid_before_utc,
    load_crt,
    load_crt_bytes,
    serialize_cert,
    verify_cert,
    verify_content,
)

JOB_AUTHORIZATION_META_KEY = "server_authorized_job"
JOB_SUBMITTER_PRINCIPAL_META_KEY = "submitter_principal"
JOB_AUTHORIZATION_SCHEMA_VERSION = 1
JOB_AUTHORIZATION_DEFAULT_TTL = 60 * 60
JOB_AUTHORIZATION_CLOCK_SKEW = 5 * 60


class JobTrustError(ValueError):
    """Raised when server-issued job authorization is missing or invalid."""


def app_data_hash(app_data: bytes) -> str:
    if not isinstance(app_data, bytes):
        raise TypeError(f"app_data must be bytes but got {type(app_data)}")
    return hashlib.sha256(app_data).hexdigest()


def canonical_manifest_json(manifest: Mapping) -> str:
    if not isinstance(manifest, Mapping):
        raise TypeError(f"manifest must be a mapping but got {type(manifest)}")
    return canonical_json(manifest)


def _server_cert_pem(id_asserter) -> str:
    cert_data = getattr(id_asserter, "cert_data", None)
    if cert_data:
        return cert_data.decode("utf-8") if isinstance(cert_data, bytes) else str(cert_data)

    cert = getattr(id_asserter, "cert", None)
    if cert:
        return serialize_cert(cert).decode("utf-8")

    raise JobTrustError("server identity asserter does not expose a certificate")


def _server_identity(id_asserter) -> str:
    cn = getattr(id_asserter, "cn", "")
    if cn:
        return str(cn)

    cert = getattr(id_asserter, "cert", None)
    if cert:
        return get_cert_common_name(cert) or ""
    return ""


def _verify_cert_validity_window(cert, verify_time: float, clock_skew: float) -> None:
    not_before = cert_not_valid_before_utc(cert).timestamp()
    not_after = cert_not_valid_after_utc(cert).timestamp()
    if verify_time + clock_skew < not_before:
        raise JobTrustError(
            f"server certificate is not yet valid: not_valid_before is {not_before - verify_time:.0f} seconds "
            f"ahead of verification time, beyond the allowed clock skew of {clock_skew:.0f} seconds"
        )
    if verify_time - clock_skew > not_after:
        raise JobTrustError(
            f"server certificate expired: not_valid_after is {verify_time - not_after:.0f} seconds "
            f"before verification time, beyond the allowed clock skew of {clock_skew:.0f} seconds"
        )


def _as_list(values) -> list:
    return list(to_tuple(values))


def _optional_float(value, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as ex:
        raise JobTrustError(f"server job authorization manifest is missing or has invalid {field_name}") from ex


def _submitter_principal(job_meta: Mapping) -> dict:
    principal = job_meta.get(JOB_SUBMITTER_PRINCIPAL_META_KEY)
    if isinstance(principal, Mapping):
        return dict(principal)
    return {}


def _manifest_submitter(job_meta: Mapping) -> dict:
    principal = _submitter_principal(job_meta)
    return {
        "subject": principal.get("subject", ""),
        "username": principal.get("username", job_meta.get(JobMetaKey.SUBMITTER_NAME.value, "")),
        "email": principal.get("email", ""),
        "org": principal.get("org", job_meta.get(JobMetaKey.SUBMITTER_ORG.value, "")),
        "effective_role": principal.get("effective_role", job_meta.get(JobMetaKey.SUBMITTER_ROLE.value, "")),
        "auth_method": principal.get("auth_method", ""),
        "issuer": principal.get("issuer", ""),
        "token_id": principal.get("token_id", ""),
    }


def issue_job_authorization(
    *,
    job_meta: Mapping,
    app_name: str,
    app_data: bytes,
    target_sites,
    id_asserter,
    issued_at: Optional[float] = None,
    expires_at: Optional[float] = None,
) -> dict:
    """Create a server-signed authorization envelope for one deployed app."""

    if not isinstance(job_meta, Mapping):
        raise TypeError(f"job_meta must be a mapping but got {type(job_meta)}")
    if not app_name:
        raise JobTrustError("app_name is required")
    if not id_asserter:
        raise JobTrustError("server identity asserter is required")

    issued = float(issued_at if issued_at is not None else time.time())
    expires = float(expires_at if expires_at is not None else issued + JOB_AUTHORIZATION_DEFAULT_TTL)
    manifest = {
        "schema_version": JOB_AUTHORIZATION_SCHEMA_VERSION,
        "job_id": job_meta.get(JobMetaKey.JOB_ID.value, ""),
        "job_name": job_meta.get(JobMetaKey.JOB_NAME.value, ""),
        "study": job_meta.get(JobMetaKey.STUDY.value, ""),
        "app_name": app_name,
        "app_sha256": app_data_hash(app_data),
        "target_sites": _as_list(target_sites),
        "submitter": _manifest_submitter(job_meta),
        "submit_time": job_meta.get(JobMetaKey.SUBMIT_TIME_ISO.value, ""),
        "server_identity": _server_identity(id_asserter),
        "issued_at": issued,
        "expires_at": expires,
    }
    signature = id_asserter.sign(canonical_manifest_json(manifest), return_str=True)
    if not signature:
        raise JobTrustError("server identity asserter did not produce a signature")

    return {
        "manifest": manifest,
        "signature": signature,
        "server_cert": _server_cert_pem(id_asserter),
    }


def attach_job_authorization(
    *,
    job_meta: dict,
    app_name: str,
    authorization: Mapping,
) -> dict:
    if not isinstance(job_meta, dict):
        raise TypeError(f"job_meta must be dict but got {type(job_meta)}")
    if not isinstance(authorization, Mapping):
        raise TypeError(f"authorization must be a mapping but got {type(authorization)}")

    per_app = job_meta.setdefault(JOB_AUTHORIZATION_META_KEY, {})
    if not isinstance(per_app, dict):
        raise JobTrustError(f"job meta key '{JOB_AUTHORIZATION_META_KEY}' must be a mapping")
    per_app[app_name] = dict(authorization)
    return job_meta


def get_job_authorization(job_meta: Mapping, app_name: str) -> Optional[dict]:
    if not isinstance(job_meta, Mapping):
        return None
    per_app = job_meta.get(JOB_AUTHORIZATION_META_KEY)
    if isinstance(per_app, Mapping):
        authz = per_app.get(app_name)
        return dict(authz) if isinstance(authz, Mapping) else None
    return None


def _verify_target(manifest: Mapping, site_name: str) -> None:
    targets = set(_as_list(manifest.get("target_sites")))
    if targets and ALL_SITES not in targets and site_name not in targets:
        raise JobTrustError(f"server job authorization does not target site '{site_name}'")


def verify_job_authorization(
    *,
    authorization: Mapping,
    app_data: bytes,
    root_ca_path: str,
    site_name: str,
    expected_job_id: str = "",
    expected_app_name: str = "",
    expected_server_identity: str = "",
    now: Optional[float] = None,
    clock_skew: float = JOB_AUTHORIZATION_CLOCK_SKEW,
) -> dict:
    """Verify a server-signed job authorization envelope and return the manifest."""

    if not isinstance(authorization, Mapping):
        raise JobTrustError("server job authorization must be a mapping")

    manifest = authorization.get("manifest")
    signature = authorization.get("signature")
    server_cert_pem = authorization.get("server_cert")
    if not isinstance(manifest, Mapping) or not signature or not server_cert_pem:
        raise JobTrustError("server job authorization is missing manifest, signature, or server_cert")

    if manifest.get("schema_version") != JOB_AUTHORIZATION_SCHEMA_VERSION:
        raise JobTrustError(f"unsupported server job authorization schema: {manifest.get('schema_version')}")
    if expected_job_id and manifest.get("job_id") != expected_job_id:
        raise JobTrustError("server job authorization job_id does not match")
    if expected_app_name and manifest.get("app_name") != expected_app_name:
        raise JobTrustError("server job authorization app_name does not match")
    if manifest.get("app_sha256") != app_data_hash(app_data):
        raise JobTrustError("server job authorization app hash does not match received app data")

    issued_at = _optional_float(manifest.get("issued_at"), "issued_at")
    expires_at = _optional_float(manifest.get("expires_at"), "expires_at")
    verify_time = float(now if now is not None else time.time())
    if issued_at > verify_time + clock_skew:
        raise JobTrustError(
            f"server job authorization was issued in the future: issued_at is "
            f"{issued_at - verify_time:.0f} seconds ahead of verification time, beyond the allowed "
            f"clock skew of {clock_skew:.0f} seconds"
        )
    if verify_time > expires_at + clock_skew:
        raise JobTrustError(
            f"server job authorization expired: expires_at is {verify_time - expires_at:.0f} seconds "
            f"before verification time, beyond the allowed clock skew of {clock_skew:.0f} seconds"
        )

    _verify_target(manifest, site_name)

    root_ca_cert = load_crt(root_ca_path)
    server_cert = load_crt_bytes(server_cert_pem.encode("utf-8"))
    verify_cert(cert_to_be_verified=server_cert, root_ca_public_key=root_ca_cert.public_key())
    _verify_cert_validity_window(server_cert, verify_time, clock_skew)
    server_identity = get_cert_common_name(server_cert) or ""
    if expected_server_identity and server_identity != expected_server_identity:
        raise JobTrustError(
            f"server job authorization identity '{server_identity}' does not match expected server "
            f"'{expected_server_identity}'"
        )
    if manifest.get("server_identity") and server_identity and manifest.get("server_identity") != server_identity:
        raise JobTrustError("server job authorization manifest identity does not match signing certificate")
    verify_content(
        content=canonical_manifest_json(manifest),
        signature=signature,
        public_key=server_cert.public_key(),
    )
    return dict(manifest)

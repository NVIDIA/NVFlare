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

from typing import Optional

from cryptography import x509
from cryptography.x509.oid import NameOID

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.net_utils import SECURE_SCHEMES
from nvflare.fuel.utils.admin_name_utils import is_valid_admin_client_name
from nvflare.fuel.utils.argument_utils import str2bool


def get_param(params: dict, key: DriverParams, default=None):
    if not params:
        return default

    value = params.get(key.value, None)
    if value is None:
        value = params.get(key, default)
    return value


def is_mtls_connection(params: dict) -> bool:
    if not params:
        return False

    scheme = get_param(params, DriverParams.SCHEME)
    secure = str2bool(get_param(params, DriverParams.SECURE, False))
    if scheme not in SECURE_SCHEMES and not secure:
        return False

    conn_security = get_param(params, DriverParams.CONNECTION_SECURITY, ConnectionSecurity.MTLS)
    return conn_security == ConnectionSecurity.MTLS


def is_mtls_config(credentials: dict, secure: bool) -> bool:
    if not secure:
        return False

    conn_security = get_param(credentials, DriverParams.CONNECTION_SECURITY, ConnectionSecurity.MTLS)
    return conn_security == ConnectionSecurity.MTLS


def get_cert_common_name(cert: x509.Certificate) -> Optional[str]:
    attrs = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    return attrs[0].value if attrs else None


def get_cert_common_name_from_pem(cert_bytes: bytes) -> Optional[str]:
    if not cert_bytes:
        return None

    cert = x509.load_pem_x509_certificate(cert_bytes)
    return get_cert_common_name(cert)


def get_cert_common_name_from_file(cert_path: str) -> Optional[str]:
    if not cert_path:
        return None

    with open(cert_path, "rb") as f:
        return get_cert_common_name_from_pem(f.read())


class CellIdentityResolver:
    def __init__(self, local_fqcn: str = "", prefix_identity_map: dict = None, exact_identity_map: dict = None):
        self.local_fqcn = FQCN.normalize(local_fqcn) if local_fqcn else ""
        self.prefix_identity_map = self._normalize_map(prefix_identity_map)
        self.exact_identity_map = self._normalize_map(exact_identity_map)

    @staticmethod
    def _normalize_map(identity_map: dict = None):
        result = {}
        if not identity_map:
            return result

        for fqcn, identity in identity_map.items():
            if fqcn and identity:
                result[FQCN.normalize(fqcn)] = identity
        return result

    @staticmethod
    def _is_descendant(fqcn: str, prefix: str) -> bool:
        return fqcn.startswith(prefix + ".")

    def _is_local_descendant_with_ancestor_prefix(self, fqcn: str, prefix: str) -> bool:
        return (
            self.local_fqcn
            and self._is_descendant(fqcn, self.local_fqcn)
            and self._is_descendant(self.local_fqcn, prefix)
        )

    def _resolve_local_child_identity(self, fqcn: str) -> Optional[str]:
        if not self.local_fqcn or not self._is_descendant(fqcn, self.local_fqcn):
            return None

        # A child connected through this local cell normally authenticates with
        # the next FQCN segment's certificate. Example: relay-a.site-1 -> site-1.
        child_suffix = fqcn[len(self.local_fqcn) + 1 :]
        parts = FQCN.split(child_suffix)
        return parts[0] if parts else None

    def resolve(self, fqcn: str) -> Optional[str]:
        if not fqcn:
            return None

        fqcn = FQCN.normalize(fqcn)
        identity = self.exact_identity_map.get(fqcn)
        if identity:
            return identity

        parts = FQCN.split(fqcn)
        for i in range(len(parts), 0, -1):
            prefix = FQCN.join(parts[:i])
            if self._is_local_descendant_with_ancestor_prefix(fqcn, prefix):
                continue

            identity = self.prefix_identity_map.get(prefix)
            if identity:
                return identity

        identity = self._resolve_local_child_identity(fqcn)
        if identity:
            return identity

        return parts[0] if parts else fqcn

    def require_match(self, fqcn: str, peer_cn: str, peer_desc: str):
        expected_cn = self.resolve(fqcn)
        if not expected_cn:
            raise ValueError(f"{peer_desc} claimed endpoint '{fqcn}' does not resolve to an expected identity")

        if not peer_cn or peer_cn == "N/A":
            raise ValueError(f"{peer_desc} does not have an authenticated mTLS peer common name")

        # Admin client cell names are per-session random IDs; the authenticated user is the cert CN.
        if is_valid_admin_client_name(fqcn):
            return

        if peer_cn != expected_cn:
            raise ValueError(
                f"{peer_desc} authenticated as '{peer_cn}' but claimed endpoint '{fqcn}' "
                f"requires identity '{expected_cn}'"
            )

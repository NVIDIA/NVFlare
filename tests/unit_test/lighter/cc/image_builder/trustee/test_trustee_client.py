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

"""Native Trustee upload authentication, failure and retirement boundaries."""

import base64
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cvm.common.contracts import resource_path
from cvm.common.errors import BuildError
from cvm.common.io import canonical, write_json
from cvm.common.policy import compose
from cvm.trustee.admin import install, retire
from cvm.trustee.client import NoRedirect, api, delete_resource, encode, upload_resource


class TrusteeClientTests(unittest.TestCase):
    def test_resource_signing_requires_explicit_resource_role_before_key_or_network_access(self):
        config = {"url": self.config["url"], "ca": "ca.pem", "admin_private_key": "secret-key-path"}
        for extra in ({}, {"admin_role": "cvm-policy"}, {"admin_role": None}, {"admin_role": "other"}):
            for operation, args in ((delete_resource, (self.resource,)), (upload_resource, (self.resource, b"a" * 64))):
                with (
                    self.subTest(extra=extra, operation=operation.__name__),
                    patch("cvm.trustee.client.Path.read_bytes") as key,
                    patch("cvm.trustee.client.urllib.request.build_opener") as opener,
                    self.assertRaisesRegex(BuildError, "admin_role=cvm-resources") as error,
                ):
                    operation(dict(config, **extra), *args)
                key.assert_not_called()
                opener.assert_not_called()
                self.assertNotIn("secret-key-path", str(error.exception))

    def test_explicit_resource_signing_and_default_policy_signing_preserve_role_separation(self):
        key = ed25519.Ed25519PrivateKey.generate()
        key_path = self.root / "issuer.key"
        key_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            )
        )
        config = {"url": self.config["url"], "ca": "ca.pem", "admin_private_key": str(key_path)}
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b""
        opener = Mock()
        opener.open.return_value = response
        with (
            patch("cvm.trustee.client.ssl.create_default_context"),
            patch("cvm.trustee.client.urllib.request.build_opener", return_value=opener),
        ):
            api(config, "GET", "resource-policy")
            delete_resource(dict(config, admin_role="cvm-resources"), self.resource)
            upload_resource(dict(config, admin_role="cvm-resources"), self.resource, b"a" * 64)
        requests = [call.args[0] for call in opener.open.call_args_list]
        self.assertEqual([request.method for request in requests], ["GET", "DELETE", "POST"])
        for request, role in zip(requests, ("cvm-policy", "cvm-resources", "cvm-resources")):
            token = request.get_header("Authorization").removeprefix("Bearer ")
            header, claims, signature = token.split(".")
            self.assertEqual(json.loads(base64.urlsafe_b64decode(claims + "=="))["role"], role)
            key.public_key().verify(base64.urlsafe_b64decode(signature + "=="), (header + "." + claims).encode())

    def test_forbidden_preissued_resource_token_reports_role_without_response_or_credentials(self):
        opener = Mock()
        opener.open.side_effect = urllib.error.HTTPError(
            "https://trustee.test", 403, "Forbidden", {}, io.BytesIO(b"secret-backend-response")
        )
        with (
            patch("cvm.trustee.client.ssl.create_default_context"),
            patch("cvm.trustee.client.urllib.request.build_opener", return_value=opener),
            self.assertRaisesRegex(BuildError, "cvm-resources token and endpoint ACL") as error,
        ):
            delete_resource(self.config, self.resource)
        self.assertNotIn("secret-backend-response", str(error.exception))
        self.assertNotIn("header.payload.signature", str(error.exception))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.token = self.root / "admin.jwt"
        self.token.write_text("header.payload.signature\n")
        self.config = {"url": "https://trustee.test:8443", "ca": "ca.pem", "admin_token_file": str(self.token)}
        self.resource = resource_path("bundle-1", "intel_tdx", bytes(32))

    def test_binary_upload_uses_native_post_and_preissued_token(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b""
        opener = Mock()
        opener.open.return_value = response
        with (
            patch("cvm.trustee.client.ssl.create_default_context"),
            patch("cvm.trustee.client.urllib.request.build_opener", return_value=opener),
        ):
            upload_resource(self.config, self.resource, b"a" * 64)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, self.config["url"] + "/kbs/v0/resource/" + self.resource)
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.data, b"a" * 64)
        self.assertEqual(request.get_header("Content-type"), "application/octet-stream")
        self.assertEqual(request.get_header("Authorization"), "Bearer header.payload.signature")
        opener.open.assert_called_once()

    def test_secret_and_resource_validation_precede_request(self):
        with patch("cvm.trustee.client.api") as request:
            for resource, secret in ((self.resource, b"short"), ("keys/bundle-1/../other", b"a" * 64)):
                with self.assertRaises(BuildError):
                    upload_resource(self.config, resource, secret)
            request.assert_not_called()

    def test_insecure_transport_ambiguous_credentials_and_token_injection_denied(self):
        with patch("cvm.trustee.client.urllib.request.build_opener") as opener:
            for config in (
                dict(self.config, url="http://trustee.test"),
                dict(self.config, admin_private_key="issuer.key"),
            ):
                with self.assertRaises(BuildError):
                    api(config, "POST", "resource/" + self.resource, b"a" * 64)
            self.token.write_text("token\nInjected: value")
            with self.assertRaises(BuildError):
                api(self.config, "POST", "resource/" + self.resource)
            opener.assert_not_called()

    def test_uncertain_upload_is_not_retried_and_does_not_disclose_data(self):
        opener = Mock()
        opener.open.side_effect = urllib.error.URLError("secret backend response")
        with (
            patch("cvm.trustee.client.ssl.create_default_context"),
            patch("cvm.trustee.client.urllib.request.build_opener", return_value=opener),
            self.assertRaises(BuildError) as error,
        ):
            upload_resource(self.config, self.resource, b"a" * 64)
        self.assertNotIn("secret backend response", str(error.exception))
        self.assertNotIn("header.payload.signature", str(error.exception))
        opener.open.assert_called_once()

    def test_redirect_does_not_forward_credentials(self):
        with self.assertRaises(BuildError):
            NoRedirect().redirect_request(None, None, 307, "redirect", {}, "https://other.test")

    def test_delete_uses_native_resource_endpoint(self):
        with patch("cvm.trustee.client.api") as request:
            delete_resource(self.config, self.resource)
        request.assert_called_once_with(self.config, "DELETE", "resource/" + self.resource)

    def test_retirement_retry_preserves_other_bundle_without_key_store_access(self):
        bundles = self.root / "bundles"
        bundles.mkdir()
        manifest = {
            "build_id": "bundle-1",
            "platform": "intel_tdx",
            "attestation_policy_id": "default",
            "measurements": {"mr_td": "1" * 96, "rtmr_0": "0" * 96, "rtmr_1": "2" * 96, "rtmr_2": "3" * 96},
        }
        other = dict(manifest, build_id="bundle-2")
        for item in (manifest, other):
            write_json(bundles / (item["build_id"] + ".json"), item)
        config = {"state": str(self.root)}
        with patch("cvm.trustee.admin.api", side_effect=BuildError("offline")), self.assertRaises(BuildError):
            retire(config, "bundle-1")
        self.assertTrue((self.root / "retired/bundle-1").is_file())
        policy = compose([other]).encode()
        with (
            patch("cvm.trustee.admin.api") as request,
            patch("cvm.trustee.admin.read_resource_policy", return_value=policy),
        ):
            retire(config, "bundle-1")
        request.assert_called_once_with(config, "POST", "resource-policy", canonical({"policy": encode(policy)}))

    def test_legacy_retirement_state_requires_migration_before_any_side_effects(self):
        state = self.root / "publisher"
        for legacy_state in (str(self.root / "legacy"), "", None):
            config = {"state": str(state), "key_service_state": legacy_state}
            for operation, args in (
                (install, (config, self.root)),
                (install, (config, self.root, True)),
                (retire, (config, "bundle-1")),
            ):
                with (
                    self.subTest(operation=operation.__name__, candidate=len(args) == 3, legacy_state=legacy_state),
                    patch("cvm.trustee.admin.verify_approval") as approval,
                    patch("cvm.trustee.admin.verify_bundle") as bundle,
                    patch("cvm.trustee.admin.api") as request,
                ):
                    with self.assertRaisesRegex(BuildError, "Migrate legacy revocations and bundle retirements"):
                        operation(*args)
                    approval.assert_not_called()
                    bundle.assert_not_called()
                    request.assert_not_called()
                    self.assertFalse(state.exists())

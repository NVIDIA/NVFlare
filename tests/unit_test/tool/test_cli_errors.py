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

from nvflare.tool.cli_errors import CLI_ERRORS, get_error


class TestGetError:
    def test_known_code_returns_tuple(self):
        message, hint = get_error("CONNECTION_FAILED", host="localhost", port=8002)
        assert isinstance(message, str)
        assert isinstance(hint, str)

    def test_placeholder_substitution(self):
        message, hint = get_error("CONNECTION_FAILED", host="myhost", port=9000)
        assert "myhost" in message
        assert "9000" in message

    def test_auth_failed(self):
        message, hint = get_error("AUTH_FAILED", username="bob")
        assert "bob" in message

    def test_timeout(self):
        message, hint = get_error("TIMEOUT", timeout=30)
        assert "30" in message
        assert "--timeout" in hint

    def test_invalid_args(self):
        message, hint = get_error("INVALID_ARGS", detail="missing --name")
        assert "missing --name" in message
        assert "--help" in hint

    def test_ca_already_exists(self):
        message, hint = get_error("CA_ALREADY_EXISTS", path="/tmp/ca")
        assert "/tmp/ca" in message
        assert "--force" in hint

    def test_ca_not_found(self):
        message, hint = get_error("CA_NOT_FOUND", ca_dir="/tmp/ca")
        assert "/tmp/ca" in message

    def test_csr_not_found(self):
        message, hint = get_error("CSR_NOT_FOUND", path="/tmp/foo.csr")
        assert "/tmp/foo.csr" in message

    def test_invalid_csr(self):
        message, hint = get_error("INVALID_CSR", path="/tmp/bad.csr")
        assert "/tmp/bad.csr" in message

    def test_cert_already_exists(self):
        message, hint = get_error("CERT_ALREADY_EXISTS", path="/tmp/cert.pem")
        assert "/tmp/cert.pem" in message

    def test_invalid_cert_type(self):
        message, hint = get_error("INVALID_CERT_TYPE", cert_type="superadmin")
        assert "superadmin" in message

    def test_key_already_exists(self):
        message, hint = get_error("KEY_ALREADY_EXISTS", path="/tmp/key.pem")
        assert "/tmp/key.pem" in message

    def test_invalid_name(self):
        message, hint = get_error("INVALID_NAME", name="bad name", reason="too long")
        assert "bad name" in message
        assert "too long" in message

    def test_cert_not_found(self):
        message, hint = get_error("CERT_NOT_FOUND", path="/tmp/cert.pem")
        assert "/tmp/cert.pem" in message

    def test_key_not_found(self):
        message, hint = get_error("KEY_NOT_FOUND", path="/tmp/key.pem")
        assert "/tmp/key.pem" in message

    def test_rootca_not_found(self):
        message, hint = get_error("ROOTCA_NOT_FOUND", path="/tmp/rootCA.pem")
        assert "/tmp/rootCA.pem" in message

    def test_invalid_endpoint(self):
        message, hint = get_error("INVALID_ENDPOINT", endpoint="badscheme://host:1234")
        assert "badscheme://host:1234" in message

    def test_output_dir_exists(self):
        message, hint = get_error("OUTPUT_DIR_EXISTS", path="/tmp/outdir")
        assert "/tmp/outdir" in message

    def test_ambiguous_key(self):
        message, hint = get_error("AMBIGUOUS_KEY", path="/tmp", files="a.key, b.key")
        assert "/tmp" in message
        assert "a.key" in message

    def test_job_not_found(self):
        message, hint = get_error("JOB_NOT_FOUND", job_id="abc-123")
        assert "abc-123" in message

    def test_unsigned_job_rejected(self):
        message, hint = get_error("UNSIGNED_JOB_REJECTED")
        assert isinstance(message, str)
        assert len(message) > 0

    def test_unknown_code_returns_fallback(self):
        message, hint = get_error("TOTALLY_UNKNOWN_CODE_XYZ")
        assert message == "Unknown error."
        assert hint == "Check logs for details."

    def test_missing_kwargs_returns_template(self):
        # If kwargs are missing for a known code, the raw template is returned gracefully
        message, hint = get_error("CONNECTION_FAILED")
        # Should contain the raw template text rather than raising
        assert "{host}" in message or "Cannot connect" in message

    def test_all_registered_codes_are_tuples(self):
        for code, value in CLI_ERRORS.items():
            assert isinstance(value, tuple), f"CLI_ERRORS[{code!r}] is not a tuple"
            assert len(value) == 2, f"CLI_ERRORS[{code!r}] does not have exactly 2 elements"
            template, hint = value
            assert isinstance(template, str), f"CLI_ERRORS[{code!r}] template is not a string"
            assert isinstance(hint, str), f"CLI_ERRORS[{code!r}] hint is not a string"

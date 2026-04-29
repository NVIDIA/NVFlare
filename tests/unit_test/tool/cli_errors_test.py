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

import pytest

from nvflare.tool.cli_errors import ERROR_REGISTRY, get_error

# --- ERROR_REGISTRY presence and format checks ---

EXPECTED_CODES = [
    "JOB_NOT_FOUND",
    "JOB_NOT_RUNNING",
    "JOB_INVALID",
    "CONNECTION_FAILED",
    "AUTH_FAILED",
    "TIMEOUT",
    "INVALID_ARGS",
    "STARTUP_KIT_MISSING",
    "SITE_NOT_FOUND",
    "LOG_CONFIG_INVALID",
    "SERVER_UNREACHABLE",
    "INTERNAL_ERROR",
    "JOB_FAILED",
    "JOB_ABORTED",
    "JOB_FINISHED_EXCEPTION",
    "JOB_ABANDONED",
]


@pytest.mark.parametrize("code", EXPECTED_CODES)
def test_all_error_codes_present(code):
    assert code in ERROR_REGISTRY


@pytest.mark.parametrize("code", EXPECTED_CODES)
def test_each_entry_has_message_and_hint(code):
    entry = ERROR_REGISTRY[code]
    assert "message" in entry
    assert "hint" in entry
    assert isinstance(entry["message"], str)
    assert isinstance(entry["hint"], str)


def test_error_registry_is_read_only():
    with pytest.raises(TypeError):
        ERROR_REGISTRY["NEW_CODE"] = {"message": "x", "hint": "y"}


def test_job_not_found_format_substitution():
    entry = ERROR_REGISTRY["JOB_NOT_FOUND"]
    result = entry["message"].format_map({"job_id": "abc123"})
    assert "abc123" in result


def test_site_not_found_format_substitution():
    entry = ERROR_REGISTRY["SITE_NOT_FOUND"]
    result = entry["message"].format_map({"site": "site-1"})
    assert "site-1" in result


def test_internal_error_hint_does_not_reference_verbose():
    assert "--verbose" not in ERROR_REGISTRY["INTERNAL_ERROR"]["hint"]


def test_startup_kit_hints_name_kit_registry_commands():
    hints = [
        ERROR_REGISTRY[code]["hint"]
        for code in ("STARTUP_KIT_MISSING", "STARTUP_KIT_NOT_CONFIGURED")
        if code in ERROR_REGISTRY
    ]
    combined = " ".join(hints)
    assert "nvflare config list" in combined
    assert "nvflare config use <id>" in combined
    assert "--kit-id <id>" in combined
    assert "--startup-kit <path>" in combined
    assert "NVFLARE_STARTUP_KIT_DIR" in combined


def test_no_error_hint_recommends_removed_startup_kit_flags():
    forbidden = [
        "--startup_kit_dir",
        "--startup_kit",
        "poc.startup_kit",
        "prod.startup_kit",
        "--{target}.startup_kit",
    ]

    offenders = {
        code: entry["hint"]
        for code, entry in ERROR_REGISTRY.items()
        if any(old_text in entry["hint"] for old_text in forbidden)
    }
    assert offenders == {}


def test_missing_substitution_key_falls_back_to_template():
    entry = ERROR_REGISTRY["JOB_NOT_FOUND"]
    try:
        entry["message"].format_map({})
        pytest.fail("Expected KeyError")
    except KeyError:
        # The spec says callers should catch KeyError and use template as-is;
        # cli_output.py handles this — just verify the template itself is untouched
        assert "{job_id}" in entry["message"]


# --- get_error() behavior checks ---


class TestGetError:
    def test_known_code_returns_tuple(self):
        message, hint = get_error("CONNECTION_FAILED")
        assert isinstance(message, str)
        assert isinstance(hint, str)

    def test_unknown_code_returns_fallback(self):
        message, hint = get_error("TOTALLY_UNKNOWN_CODE_XYZ")
        assert message == "Unknown error."
        assert hint == "Check logs for details."

    def test_unknown_code_raises_in_dev_mode(self, monkeypatch):
        monkeypatch.setenv("NVFLARE_DEV", "1")
        with pytest.raises(KeyError):
            get_error("TOTALLY_UNKNOWN_CODE_XYZ")

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

    def test_all_registered_codes_have_string_message_and_hint(self):
        for code, entry in ERROR_REGISTRY.items():
            assert isinstance(entry["message"], str), f"ERROR_REGISTRY[{code!r}] message is not a string"
            assert isinstance(entry["hint"], str), f"ERROR_REGISTRY[{code!r}] hint is not a string"

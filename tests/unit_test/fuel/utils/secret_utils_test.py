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

import time
import warnings

import pytest

from nvflare.fuel.utils.secret_utils import (
    PotentialSecretWarning,
    UnsupportedSecretRefWarning,
    find_potential_secrets,
    has_secret_refs,
    resolve_secret_refs,
    secret_file_ref,
    secret_ref,
    split_command_preserving_secret_refs,
    warn_on_potential_secrets,
    warn_on_unsupported_secret_ref_keys,
    warn_on_unsupported_secret_refs,
    warn_on_unsupported_secret_refs_outside_keys,
)

# Fake credentials for testing the detector -- none of these are real.
FAKE_GITHUB_TOKEN = "ghp_" + "Ab1" * 12
FAKE_AWS_KEY_ID = "AKIA" + "ABCDEFGHIJKLMNOP"
FAKE_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N3XgL0n3I9P"
FAKE_HEX_KEY = "0123456789abcdef0123456789abcdef01234567"
FAKE_HIGH_ENTROPY = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789+/=-_"


class TestSecretRef:
    def test_secret_ref_format(self):
        assert secret_ref("MY_API_KEY") == "${secret:MY_API_KEY}"

    @pytest.mark.parametrize("bad_name", ["", "1BAD", "MY-KEY", "MY KEY", None, 123])
    def test_secret_ref_invalid_name_raises(self, bad_name):
        with pytest.raises(ValueError):
            secret_ref(bad_name)

    def test_secret_file_ref_format(self):
        assert secret_file_ref("/var/run/secrets/my-app/api-key") == "${secret:file:/var/run/secrets/my-app/api-key}"

    @pytest.mark.parametrize(
        "bad_path",
        [
            "",
            "/bad{path",
            "/bad}path",
            "/secret path/key",
            "/bad\npath",
            "/bad\0path",
            "/bad\x1bpath",
            None,
            123,
        ],
    )
    def test_secret_file_ref_invalid_path_raises(self, bad_path):
        with pytest.raises(ValueError):
            secret_file_ref(bad_path)

    def test_has_secret_refs(self):
        assert has_secret_refs("--key ${secret:MY_KEY}")
        assert has_secret_refs("--key-file ${secret:file:/var/run/secrets/my-app/api-key}")
        assert not has_secret_refs("--key plainvalue")
        assert not has_secret_refs(123)

    def test_resolve_single_ref(self):
        assert resolve_secret_refs("--key ${secret:A}", env={"A": "value-a"}) == "--key value-a"

    def test_resolve_multiple_refs(self):
        result = resolve_secret_refs("${secret:A}:${secret:B}", env={"A": "x", "B": "y z"})
        assert result == "x:y z"

    def test_resolve_file_ref(self, tmp_path):
        secret_file = tmp_path / "api-key"
        secret_file.write_text("file-secret-value\n")

        assert resolve_secret_refs(secret_file_ref(str(secret_file)), env={}) == "file-secret-value"

    def test_resolve_env_and_file_refs_recursively_without_resolving_keys(self, tmp_path):
        secret_file = tmp_path / "api-key"
        secret_file.write_text("file-secret-value")
        unresolved_key = "${secret:KEY_MUST_NOT_BE_RESOLVED}"
        value = {
            "env": "prefix-${secret:API_KEY}",
            "nested": [secret_file_ref(str(secret_file)), {"pair": ("${secret:A}", "plain")}],
            unresolved_key: "${secret:B}",
        }

        result = resolve_secret_refs(
            value,
            env={"API_KEY": "env-secret-value", "A": "a-secret-value", "B": "b-secret-value"},
        )

        assert result == {
            "env": "prefix-env-secret-value",
            "nested": ["file-secret-value", {"pair": ("a-secret-value", "plain")}],
            unresolved_key: "b-secret-value",
        }

    def test_resolve_no_refs_passthrough(self):
        assert resolve_secret_refs("--epochs 5", env={}) == "--epochs 5"

    @pytest.mark.parametrize("posix", [False, True])
    def test_command_tokenization_preserves_reference_characters(self, posix):
        reference = "${secret:file:/tmp/api'key\\value}"
        tokens = split_command_preserving_secret_refs(f"python train.py --api-key {reference}", posix=posix)

        assert tokens == ["python", "train.py", "--api-key", reference]

    def test_command_tokenization_preserves_malformed_reference_for_validation(self):
        malformed = "${secret:BAD\\NAME}"
        tokens = split_command_preserving_secret_refs(f"--api-key {malformed}", posix=True)

        assert tokens == ["--api-key", malformed]
        with pytest.raises(ValueError, match="invalid secret reference syntax"):
            resolve_secret_refs(tokens[1], env={"BADNAME": "must-not-resolve"})

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            (
                '--publisher O\'Reilly --authorization "Bearer ${secret:TOKEN}"',
                ["--publisher", "O'Reilly", "--authorization", "Bearer ${secret:TOKEN}"],
            ),
            (
                "--note unmatched\"quote --authorization 'Bearer ${secret:TOKEN}'",
                ["--note", 'unmatched"quote', "--authorization", "Bearer ${secret:TOKEN}"],
            ),
            (
                '--note unmatched"quote --authorization "Bearer ${secret:TOKEN}"',
                ["--note", 'unmatched"quote', "--authorization", "Bearer ${secret:TOKEN}"],
            ),
            (
                '--authorization="Bearer ${secret:TOKEN}"',
                ["--authorization=Bearer ${secret:TOKEN}"],
            ),
            (
                '--authorization=prefix"Bearer ${secret:TOKEN}"',
                ["--authorization=prefixBearer ${secret:TOKEN}"],
            ),
        ],
    )
    def test_command_tokenization_finds_quoted_secret_ref_after_unmatched_quote(self, command, expected):
        tokens = split_command_preserving_secret_refs(command, posix=False, group_secret_ref_quotes=True)

        assert tokens == expected

    def test_command_tokenization_preserves_balanced_non_secret_quotes_around_unquoted_ref(self):
        for quoted in ('"one"', '"one "', '"one:"'):
            command = f'--a {quoted} --b ${{secret:TOKEN}} --c "two"'
            tokens = split_command_preserving_secret_refs(
                command,
                posix=False,
                group_secret_ref_quotes=True,
            )

            assert tokens == command.split()

    def test_resolve_missing_env_var_raises_without_leaking(self):
        with pytest.raises(ValueError) as exc_info:
            resolve_secret_refs("--key ${secret:MISSING_VAR}", env={"OTHER": "other-value"})
        assert "MISSING_VAR" in str(exc_info.value)
        assert "other-value" not in str(exc_info.value)

    def test_resolve_missing_file_raises_without_leaking_other_values(self, tmp_path):
        missing_file = tmp_path / "missing-api-key"
        with pytest.raises(ValueError) as exc_info:
            resolve_secret_refs(secret_file_ref(str(missing_file)), env={"OTHER": "other-secret-value"})
        assert str(missing_file) in str(exc_info.value)
        assert "other-secret-value" not in str(exc_info.value)

    @pytest.mark.parametrize(
        "malformed_ref",
        ["${secret:BAD-NAME}", "${secret:UNFINISHED", "${secret:file:/secret path/key}"],
    )
    def test_resolve_malformed_ref_raises_without_echoing_value(self, malformed_ref):
        with pytest.raises(ValueError) as exc_info:
            resolve_secret_refs(malformed_ref, env={})

        assert "invalid secret reference syntax" in str(exc_info.value)
        assert malformed_ref not in str(exc_info.value)


class TestFindPotentialSecrets:
    @pytest.mark.parametrize(
        "value",
        [
            f"--api_key {FAKE_GITHUB_TOKEN}",
            f"--aws-key {FAKE_AWS_KEY_ID}",
            FAKE_JWT,
            "-----BEGIN PRIVATE KEY-----\nMIIabcdef\n-----END PRIVATE KEY-----",
            "-----BEGIN RSA PRIVATE KEY-----",
            "--tracking_uri https://alice:p4ssw0rd123@tracking.example.com",
            "--password hunter22x",
            "--auth-token=abcd1234efgh",
            "env API_PASSWORD=hunter22x python3 -u",
            "API_TOKEN=actualSecret123",
            "docker run --env=API_PASSWORD=hunter22x image",
            "kubectl run app --env=API_TOKEN=actualSecret123",
            'powershell -Command "$env:API_PASSWORD=hunter22x"',
            "sh -c 'export API_PASSWORD=hunter22x; python3 train.py'",
            "sh -c 'MODE=prod API_PASSWORD=hunter22x python3 train.py'",
            'powershell -Command "$env:MODE=prod; $env:API_PASSWORD=hunter22x; python train.py"',
            "curl 'https://example.test?mode=prod&API_TOKEN=hunter22x'",
            "MODE=prod,API_PASSWORD=hunter22x",
            "MODE=prod&API_PASSWORD=hunter22x",
            "env API_PASSWORD=?hunter22x python3",
            "env API_PASSWORD=&hunter22x python3",
            "env API_PASSWORD=,hunter22x python3",
            "env API_PASSWORD=abc?hunter22x python3",
            'API_TOKEN="sv=2024-01-01&sig=abcdefgh123456"',
            'API_PASSWORD="abc MODE=hunter22x"',
            f"--wandb_key {FAKE_HEX_KEY}",
            FAKE_HIGH_ENTROPY,
            {"api_key": "abcd1234efgh"},
            {"outer": {"client_secret": "abcd1234efgh"}},
            {"receivers": [{"password": "abcd1234efgh"}]},
            {"password": "$uperSecret123"},
            {"Authorization": "Bearer abcdefgh123456"},
        ],
    )
    def test_positives(self, value):
        assert find_potential_secrets(value, location="test")

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "--epochs 10 --lr 0.001 --batch_size 32",
            "--tokenizer_path /models/tok --max_tokens 512",
            "--data_path /data/site-1/train.csv",
            "--key_metric accuracy",
            "--use_auth true",
            "https://mlflow.example.com/api/2.0/experiments",
            "/opt/very/long/path/to/experiment/data_v2/train",
            f"--api-key {secret_ref('MY_KEY')}",
            "env API_TOKEN=${secret:API_TOKEN} python3 -u",
            "API_TOKEN=${secret:API_TOKEN}&MODE=prod",
            "API_PASSWORD=$API_PASSWORD python3 -u",
            "docker run --env=API_TOKEN=${secret:API_TOKEN} image",
            'powershell -Command "$env:API_PASSWORD=${secret:API_PASSWORD}"',
            "sh -c 'export API_PASSWORD=${secret:API_PASSWORD}; python3 train.py'",
            'powershell -Command "$env:MODE=prod; $env:API_PASSWORD=${secret:API_PASSWORD}; python train.py"',
            'AUTHORIZATION="Bearer ${secret:AUTH_TOKEN}" python3 -u',
            "env DATA_PATH=/data/site-1/train.csv python3 -u",
            {"token": "${secret:MY_TOKEN}"},
            {"token": "${secret:file:/var/run/secrets/my-app/api-key}"},
            {"private_key": "/etc/secrets/key.pem"},
            {"password": "$PASSWORD_VAR"},
            {"password": "${PASSWORD_VAR}"},
            {"password": "%PASSWORD_VAR%"},
            {"token": "{JOB_ID}"},
            {"epochs": 10, "lr": 0.001},
            ["--epochs", "10"],
            42,
            None,
        ],
    )
    def test_negatives(self, value):
        assert find_potential_secrets(value, location="test") == []

    def test_finding_reports_location(self):
        findings = find_potential_secrets({"site-1": {"api_key": "abcd1234efgh"}}, location="per_site_config")
        assert len(findings) == 1
        assert "per_site_config" in findings[0].location
        assert findings[0].reason == "value of a secret-named key"

    def test_findings_never_contain_the_secret(self):
        secret_value = "abcdefgh1234567890secret"
        findings = find_potential_secrets({"password": secret_value}, location="test")
        assert findings
        for finding in findings:
            assert secret_value not in finding.preview
            assert secret_value not in finding.location
            assert secret_value not in finding.reason

    def test_secret_like_mapping_key_is_masked_in_all_findings(self):
        findings = find_potential_secrets(
            {FAKE_GITHUB_TOKEN: {"password": "abcdefgh1234"}},
            location="test",
        )
        assert findings
        for finding in findings:
            assert FAKE_GITHUB_TOKEN not in finding.preview
            assert FAKE_GITHUB_TOKEN not in finding.location
            assert FAKE_GITHUB_TOKEN not in finding.reason

    def test_low_entropy_mapping_key_is_not_copied_into_descendant_finding(self):
        key_secret = "hunter22x"
        findings = find_potential_secrets({key_secret: {"password": "abcdefgh1234"}}, location="test")
        assert findings
        assert all(key_secret not in finding.location for finding in findings)

    def test_secret_like_flag_name_is_masked_in_all_findings(self):
        secret_flag = f"--api-key-{FAKE_GITHUB_TOKEN}"
        findings = find_potential_secrets(f"{secret_flag} dummyvalue", location="test")
        assert findings
        for finding in findings:
            assert FAKE_GITHUB_TOKEN not in finding.preview
            assert FAKE_GITHUB_TOKEN not in finding.location
            assert FAKE_GITHUB_TOKEN not in finding.reason

    def test_findings_deduplicated(self):
        # the same token as both flag value and known format must yield one finding per location
        findings = find_potential_secrets(f"--github_token {FAKE_GITHUB_TOKEN}", location="test")
        previews = [f.preview for f in findings]
        assert len(previews) == len(set((f.location, f.preview) for f in findings))

    @pytest.mark.parametrize(
        "value",
        [
            {"password": "${secret:BAD-NAME}"},
            {"password": "${secret:KEY}hunter22x"},
        ],
    )
    def test_malformed_or_composite_refs_do_not_bypass_detection(self, value):
        assert find_potential_secrets(value, location="test")

    def test_authorization_scheme_with_valid_ref_is_safe(self):
        assert find_potential_secrets({"auth_token": "Bearer ${secret:AUTH_TOKEN}"}, location="test") == []

    @pytest.mark.parametrize("quote", ['"', "'"])
    def test_unterminated_quoted_assignment_with_backslashes_scans_safely(self, quote):
        value = f"API_PASSWORD={quote}" + "\\" * 256

        start = time.monotonic()
        find_potential_secrets(value, location="test")
        assert time.monotonic() - start < 1.0

    def test_unterminated_quoted_flag_value_is_scanned_as_one_value(self):
        findings = find_potential_secrets('--password "short secret value', location="test")

        assert any(finding.reason == "value of a secret-named flag" for finding in findings)

    def test_unterminated_quote_before_flag_does_not_hide_flag(self):
        findings = find_potential_secrets('"legacy --password hunter22x', location="test")

        assert any(finding.reason == "value of a secret-named flag" for finding in findings)


class TestWarnOnPotentialSecrets:
    def test_warns_with_context_and_no_leak(self):
        with pytest.warns(PotentialSecretWarning) as record:
            findings = warn_on_potential_secrets(f"--api_key {FAKE_GITHUB_TOKEN}", context="train_args")
        assert findings
        messages = [str(w.message) for w in record]
        assert any("train_args" in m for m in messages)
        assert all(FAKE_GITHUB_TOKEN not in m for m in messages)
        assert all(w.filename == "<nvflare-secret-scan>" for w in record)

    def test_no_warning_for_clean_value(self):
        import warnings as warnings_module

        with warnings_module.catch_warnings(record=True) as record:
            warnings_module.simplefilter("always")
            findings = warn_on_potential_secrets("--epochs 5 --lr 0.1", context="train_args")
        assert findings == []
        assert not [w for w in record if issubclass(w.category, PotentialSecretWarning)]

    def test_warning_as_error_is_safely_reemitted_without_raising(self):
        secret_value = "actualSecret123"
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("error", PotentialSecretWarning)
            findings = warn_on_potential_secrets(
                {"password": secret_value},
                context="add_client_config config",
            )

        assert findings
        assert len(record) == 1
        assert record[0].filename == "<nvflare-secret-scan>"
        assert secret_value not in str(record[0].message)


class TestUnsupportedSecretRefs:
    def test_warns_without_including_reference(self):
        reference = secret_ref("MY_API_KEY")
        with pytest.warns(UnsupportedSecretRefWarning) as record:
            assert warn_on_unsupported_secret_refs(
                {"nested": [reference]},
                context="recipe parameter 'task_data'",
            )

        assert all(reference not in str(w.message) for w in record)
        assert all(w.filename == "<nvflare-secret-scan>" for w in record)

    def test_no_warning_without_reference(self):
        assert not warn_on_unsupported_secret_refs({"nested": ["plain"]}, context="test")

    def test_warns_for_nested_mapping_key_but_not_value(self):
        reference = secret_ref("MY_API_KEY")
        with pytest.warns(UnsupportedSecretRefWarning):
            assert warn_on_unsupported_secret_ref_keys(
                {"outer": [{reference: "value"}]},
                context="add_client_config config",
            )
        assert not warn_on_unsupported_secret_ref_keys({"outer": [{"token": reference}]}, context="test")

    def test_warns_only_outside_supported_mapping_values(self):
        reference = secret_ref("MY_API_KEY")
        assert not warn_on_unsupported_secret_refs_outside_keys(
            {"site-1": {"train_args": f"--api-key {reference}"}},
            supported_value_keys={"train_args"},
            supported_value_depth=2,
            context="per_site_config",
        )
        with pytest.warns(UnsupportedSecretRefWarning):
            assert warn_on_unsupported_secret_refs_outside_keys(
                {"site-1": {"subsection": {"train_args": reference}}},
                supported_value_keys={"train_args"},
                supported_value_depth=2,
                context="per_site_config",
            )
        with pytest.warns(UnsupportedSecretRefWarning):
            assert warn_on_unsupported_secret_refs_outside_keys(
                {"site-1": {"train_args": {"nested": reference}}},
                supported_value_keys={"train_args"},
                supported_value_depth=2,
                context="per_site_config",
            )
        with pytest.warns(UnsupportedSecretRefWarning, match="train_args"):
            assert warn_on_unsupported_secret_refs_outside_keys(
                {"site-1": {"train_script": reference}},
                supported_value_keys={"train_args"},
                supported_value_depth=2,
                context="per_site_config",
            )

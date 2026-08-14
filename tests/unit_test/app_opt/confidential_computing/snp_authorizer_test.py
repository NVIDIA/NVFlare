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

import base64
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

from nvflare.app_opt.confidential_computing.cc_authorizer import CCTokenGenerateError
from nvflare.app_opt.confidential_computing.snp_authorizer import (
    CHIP_ID_OFFSET,
    CHIP_ID_SIZE,
    MIN_REPORT_SIZE,
    REPORT_DATA_OFFSET,
    REPORT_DATA_SIZE,
    REPORT_SIZE,
    REPORTED_TCB_OFFSET,
    REPORTED_TCB_SIZE,
    SNPAuthorizer,
)

CHALLENGE = "ab" * 32
OTHER_CHALLENGE = "cd" * 32


def _report(challenge=CHALLENGE, chip_id=b"c" * CHIP_ID_SIZE):
    report = bytearray(MIN_REPORT_SIZE)
    report_data = bytes.fromhex(challenge).ljust(REPORT_DATA_SIZE, b"\0")
    report[REPORT_DATA_OFFSET : REPORT_DATA_OFFSET + REPORT_DATA_SIZE] = report_data
    report[REPORTED_TCB_OFFSET : REPORTED_TCB_OFFSET + REPORTED_TCB_SIZE] = b"t" * REPORTED_TCB_SIZE
    report[CHIP_ID_OFFSET : CHIP_ID_OFFSET + CHIP_ID_SIZE] = chip_id
    return bytes(report)


def test_generate_uses_private_temporary_files_and_returns_text(tmp_path):
    authorizer = SNPAuthorizer(amd_certs_dir=str(tmp_path), max_retries=1, retry_interval=0)

    def run(cmd, _action):
        assert cmd[:2] == ["snpguest", "report"]
        with open(cmd[3], "rb") as request_stream:
            report_data = request_stream.read()
        assert report_data == bytes.fromhex(CHALLENGE).ljust(REPORT_DATA_SIZE, b"\0")
        with open(cmd[2], "wb") as report_stream:
            report_stream.write(_report())
        return subprocess.CompletedProcess(cmd, 0, "", "")

    authorizer._run_with_retry = run
    token = authorizer.generate(CHALLENGE)

    assert isinstance(token, str)
    assert base64.b64decode(token)[REPORT_DATA_OFFSET : REPORT_DATA_OFFSET + REPORT_DATA_SIZE]
    assert not os.path.exists("request.bin")
    assert not os.path.exists("report.bin")


def test_generate_raises_cc_error_when_snpguest_fails(tmp_path):
    authorizer = SNPAuthorizer(amd_certs_dir=str(tmp_path), max_retries=1, retry_interval=0)
    authorizer._run_with_retry = Mock(side_effect=RuntimeError("device unavailable"))

    with pytest.raises(CCTokenGenerateError, match="device unavailable"):
        authorizer.generate(CHALLENGE)


def test_generate_rejects_report_with_non_abi_size(tmp_path):
    authorizer = SNPAuthorizer(amd_certs_dir=str(tmp_path), max_retries=1, retry_interval=0)

    def run(cmd, _action):
        with open(cmd[2], "wb") as report_stream:
            report_stream.write(b"x" * (REPORT_SIZE - 1))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    authorizer._run_with_retry = run

    with pytest.raises(CCTokenGenerateError, match="invalid size"):
        authorizer.generate(CHALLENGE)


def test_verify_checks_certificate_chain_signature_and_challenge(tmp_path):
    (tmp_path / "ark.pem").write_text("ark")
    (tmp_path / "ask.pem").write_text("ask")
    authorizer = SNPAuthorizer(amd_certs_dir=str(tmp_path), max_retries=1, retry_interval=0)
    report = _report()
    expected_cache_key = f"{'63' * CHIP_ID_SIZE}-{'74' * REPORTED_TCB_SIZE}"
    (tmp_path / f"vcek-{expected_cache_key}.pem").write_text("vcek")
    calls = []

    def run(cmd, action):
        calls.append((cmd, action))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    authorizer._run_with_retry = run
    authorizer._run_once = run
    token = base64.b64encode(report).decode("ascii")

    assert authorizer.verify(token, CHALLENGE) is True
    verify_calls = [cmd for cmd, action in calls if action == "verify_attestation"]
    cert_calls = [cmd for cmd, action in calls if action == "verify_cert_chain"]
    assert len(cert_calls) == 1
    assert cert_calls[0] == ["snpguest", "verify", "certs", os.path.dirname(verify_calls[0][4])]
    assert len(verify_calls) == 1
    assert verify_calls[0][0:3] == ["snpguest", "verify", "attestation"]
    assert os.path.dirname(verify_calls[0][4]) == verify_calls[0][3]


def test_verify_rejects_mismatched_verifier_challenge_before_commands(tmp_path):
    authorizer = SNPAuthorizer(amd_certs_dir=str(tmp_path), max_retries=1, retry_interval=0)
    authorizer._run_with_retry = Mock()
    authorizer._run_once = Mock()
    token = base64.b64encode(_report()).decode("ascii")

    assert authorizer.verify(token, OTHER_CHALLENGE) is False
    authorizer._run_with_retry.assert_not_called()
    authorizer._run_once.assert_not_called()


def test_verify_rejects_missing_verifier_challenge(tmp_path):
    authorizer = SNPAuthorizer(amd_certs_dir=str(tmp_path), max_retries=1, retry_interval=0)
    authorizer._run_with_retry = Mock()
    authorizer._run_once = Mock()

    assert authorizer.verify(base64.b64encode(_report()).decode("ascii")) is False
    authorizer._run_with_retry.assert_not_called()
    authorizer._run_once.assert_not_called()


def test_verify_fetches_and_reuses_certificate_cache(tmp_path):
    authorizer = SNPAuthorizer(amd_certs_dir=str(tmp_path), max_retries=1, retry_interval=0)
    actions = []

    def run(cmd, action):
        actions.append(action)
        if action == "fetch_ca_certs":
            (tmp_path / "ark.pem").write_text("ark")
            (tmp_path / "ask.pem").write_text("ask")
        elif action == "fetch_vcek":
            with open(os.path.join(cmd[4], "vcek.pem"), "w") as vcek_stream:
                vcek_stream.write("vcek")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    authorizer._run_with_retry = run
    authorizer._run_once = run
    first = base64.b64encode(_report(CHALLENGE)).decode("ascii")
    second = base64.b64encode(_report(OTHER_CHALLENGE)).decode("ascii")

    assert authorizer.verify(first, CHALLENGE) is True
    assert authorizer.verify(second, OTHER_CHALLENGE) is True
    assert actions.count("fetch_ca_certs") == 1
    assert actions.count("fetch_vcek") == 1
    assert actions.count("verify_cert_chain") == 2
    assert actions.count("verify_attestation") == 2


def test_ca_certificate_fetch_is_serialized_across_concurrent_authorizers(tmp_path):
    authorizers = [
        SNPAuthorizer(amd_certs_dir=str(tmp_path), max_retries=1, retry_interval=0),
        SNPAuthorizer(amd_certs_dir=str(tmp_path), max_retries=1, retry_interval=0),
    ]
    state_lock = threading.Lock()
    start = threading.Barrier(2)
    fetch_count = 0
    active = 0
    max_active = 0

    def run(cmd, action):
        nonlocal active, fetch_count, max_active
        assert action == "fetch_ca_certs"
        with state_lock:
            fetch_count += 1
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.05)
            (tmp_path / "ark.pem").write_text("ark")
            (tmp_path / "ask.pem").write_text("ask")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        finally:
            with state_lock:
                active -= 1

    for authorizer in authorizers:
        authorizer._run_with_retry = run

    def ensure(authorizer):
        start.wait()
        authorizer._ensure_amd_ca_certs()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [executor.submit(ensure, authorizer) for authorizer in authorizers]
        for result in results:
            result.result()

    assert fetch_count == 1
    assert max_active == 1


@pytest.mark.parametrize(
    "token",
    [
        "",
        "not-base64!",
        base64.b64encode(b"short").decode("ascii"),
        base64.b64encode(b"x" * (REPORT_SIZE + 1)).decode("ascii"),
        base64.b64encode(b"x" * (REPORT_SIZE * 100)).decode("ascii"),
        None,
    ],
)
def test_verify_rejects_malformed_reports_without_running_commands(tmp_path, token):
    authorizer = SNPAuthorizer(amd_certs_dir=str(tmp_path), max_retries=1, retry_interval=0)
    authorizer._run_with_retry = Mock()
    authorizer._run_once = Mock()

    assert authorizer.verify(token, CHALLENGE) is False
    authorizer._run_with_retry.assert_not_called()
    authorizer._run_once.assert_not_called()


def test_verify_does_not_retry_deterministic_signature_failure(tmp_path):
    (tmp_path / "ark.pem").write_text("ark")
    (tmp_path / "ask.pem").write_text("ask")
    cache_key = f"{'63' * CHIP_ID_SIZE}-{'74' * REPORTED_TCB_SIZE}"
    (tmp_path / f"vcek-{cache_key}.pem").write_text("vcek")
    authorizer = SNPAuthorizer(amd_certs_dir=str(tmp_path), max_retries=5, retry_interval=10)

    with (
        patch(
            "subprocess.run",
            side_effect=[
                subprocess.CompletedProcess([], 0, "", ""),
                subprocess.CompletedProcess([], 2, "", "signature verification failed"),
            ],
        ) as run,
        patch("nvflare.app_opt.confidential_computing.snp_authorizer.time.sleep") as sleep,
    ):
        assert authorizer.verify(base64.b64encode(_report()).decode("ascii"), CHALLENGE) is False

    assert run.call_count == 2
    sleep.assert_not_called()


def test_vcek_fetch_is_not_retried_for_report_derived_failure(tmp_path):
    (tmp_path / "ark.pem").write_text("ark")
    (tmp_path / "ask.pem").write_text("ask")
    authorizer = SNPAuthorizer(amd_certs_dir=str(tmp_path), max_retries=5, retry_interval=10, vcek_fetch_interval=0)

    with (
        patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess([], 2, "", "KDS rejected report fields"),
        ) as run,
        patch("nvflare.app_opt.confidential_computing.snp_authorizer.time.sleep") as sleep,
    ):
        assert authorizer.verify(base64.b64encode(_report()).decode("ascii"), CHALLENGE) is False

    run.assert_called_once()
    sleep.assert_not_called()


def test_vcek_cache_limit_rejects_new_entries_without_fetch(tmp_path):
    (tmp_path / "vcek-existing.pem").write_text("vcek")
    authorizer = SNPAuthorizer(amd_certs_dir=str(tmp_path), max_vcek_cache_entries=1, max_retries=1, retry_interval=0)
    authorizer._run_once = Mock()

    with pytest.raises(RuntimeError, match="cache entry limit"):
        authorizer._ensure_amd_vcek("new", str(tmp_path / "report.bin"))

    authorizer._run_once.assert_not_called()


def test_verify_rejects_untrusted_vcek_chain_before_attestation(tmp_path):
    (tmp_path / "ark.pem").write_text("ark")
    (tmp_path / "ask.pem").write_text("ask")
    cache_key = f"{'63' * CHIP_ID_SIZE}-{'74' * REPORTED_TCB_SIZE}"
    (tmp_path / f"vcek-{cache_key}.pem").write_text("untrusted-vcek")
    authorizer = SNPAuthorizer(amd_certs_dir=str(tmp_path), max_retries=1, retry_interval=0)
    actions = []

    def run(_cmd, action):
        actions.append(action)
        if action == "verify_cert_chain":
            raise RuntimeError("certificate chain verification failed")
        return subprocess.CompletedProcess([], 0, "", "")

    authorizer._run_once = run

    assert authorizer.verify(base64.b64encode(_report()).decode("ascii"), CHALLENGE) is False
    assert actions == ["verify_cert_chain"]


def test_consecutive_distinct_vcek_fetches_wait_instead_of_rejecting(tmp_path):
    authorizer = SNPAuthorizer(
        amd_certs_dir=str(tmp_path),
        max_retries=1,
        retry_interval=0,
        vcek_fetch_interval=0.01,
    )
    actions = []

    def run(cmd, action):
        actions.append(action)
        if action == "fetch_ca_certs":
            (tmp_path / "ark.pem").write_text("ark")
            (tmp_path / "ask.pem").write_text("ask")
        elif action == "fetch_vcek":
            with open(os.path.join(cmd[4], "vcek.pem"), "w") as vcek_stream:
                vcek_stream.write("vcek")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    authorizer._run_with_retry = run
    authorizer._run_once = run
    first = base64.b64encode(_report(CHALLENGE, b"a" * CHIP_ID_SIZE)).decode("ascii")
    second = base64.b64encode(_report(OTHER_CHALLENGE, b"b" * CHIP_ID_SIZE)).decode("ascii")

    assert authorizer.verify(first, CHALLENGE) is True
    assert authorizer.verify(second, OTHER_CHALLENGE) is True
    assert actions.count("fetch_vcek") == 2

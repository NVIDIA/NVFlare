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

import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from nvflare.app_opt.confidential_computing.cc_authorizer import CCTokenGenerateError
from nvflare.app_opt.confidential_computing.gpu_authorizer import GPUAuthorizer


def _evidence(nonce="a" * 64):
    return {
        "evidences": [{"arch": "HOPPER", "nonce": nonce, "evidence": "signed", "certificate": "chain"}],
        "result_code": 0,
        "result_message": "Ok",
    }


def _verification(nonce="a" * 64):
    return {
        "claims": [
            {
                "eat_nonce": nonce,
                "x-nvidia-device-type": "gpu",
                "measres": "success",
                "secboot": True,
                "dbgstat": "disabled",
                "x-nvidia-gpu-arch-check": True,
                "x-nvidia-gpu-attestation-report-parsed": True,
                "x-nvidia-gpu-attestation-report-nonce-match": True,
                "x-nvidia-gpu-attestation-report-signature-verified": True,
                "x-nvidia-gpu-attestation-report-cert-chain": {
                    "x-nvidia-cert-status": "valid",
                    "x-nvidia-cert-ocsp-status": "good",
                },
                "x-nvidia-gpu-attestation-report-cert-chain-fwid-match": True,
                "x-nvidia-gpu-driver-rim-signature-verified": True,
                "x-nvidia-gpu-driver-rim-version-match": True,
                "x-nvidia-gpu-vbios-rim-signature-verified": True,
                "x-nvidia-gpu-vbios-rim-version-match": True,
                "x-nvidia-mismatch-measurement-records": None,
            }
        ],
        "result_code": 0,
        "result_message": "Ok",
    }


class FakeRunner:
    def __init__(self):
        self.calls = []
        self.verify_evidence = None
        self.collect_result = subprocess.CompletedProcess([], 0, json.dumps(_evidence()))
        self.verify_result = subprocess.CompletedProcess([], 0, json.dumps(_verification()))

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if "collect-evidence" in command:
            nonce = command[command.index("--nonce") + 1]
            result = json.loads(self.collect_result.stdout)
            if self.collect_result.returncode == 0 and result.get("evidences"):
                for evidence in result["evidences"]:
                    evidence["nonce"] = nonce
            return subprocess.CompletedProcess(command, self.collect_result.returncode, json.dumps(result))
        evidence_path = command[command.index("--gpu-evidence-file") + 1]
        with open(evidence_path, encoding="utf-8") as evidence_stream:
            self.verify_evidence = json.load(evidence_stream)
        return self.verify_result


def test_generate_collects_nonce_bound_nvat_evidence(monkeypatch):
    runner = FakeRunner()
    monkeypatch.setattr("subprocess.run", runner)
    monkeypatch.setenv("NV_ATTESTATION_SERVICE_KEY", "test-service-key")
    authorizer = GPUAuthorizer(nvat_command="nvattest")

    token = json.loads(authorizer.generate())

    assert token["result_code"] == 0
    assert len(token["evidences"][0]["nonce"]) == 64
    command, kwargs = runner.calls[0]
    assert command[:6] == ["nvattest", "--format", "json", "--log-level", "warn", "collect-evidence"]
    assert kwargs["env"]["NV_ATTESTATION_SERVICE_KEY"] == "test-service-key"
    assert "test-service-key" not in command
    assert authorizer._can_generate is True


def test_explicit_service_key_precedes_environment(monkeypatch):
    runner = FakeRunner()
    monkeypatch.setattr("subprocess.run", runner)
    monkeypatch.setenv("NV_ATTESTATION_SERVICE_KEY", "environment-key")
    authorizer = GPUAuthorizer(nvat_command="nvattest", service_key="constructor-key")

    authorizer.generate()

    _, kwargs = runner.calls[0]
    assert kwargs["env"]["NV_ATTESTATION_SERVICE_KEY"] == "constructor-key"


def test_generate_fails_closed_on_nvat_error(monkeypatch):
    runner = FakeRunner()
    runner.collect_result = subprocess.CompletedProcess([], 7, "{}")
    monkeypatch.setattr("subprocess.run", runner)
    authorizer = GPUAuthorizer(nvat_command="nvattest")

    with pytest.raises(CCTokenGenerateError, match="exit status 7"):
        authorizer.generate()
    assert authorizer._can_generate is False


def test_verify_appraises_file_evidence_before_recording_nonce(monkeypatch):
    runner = FakeRunner()
    monkeypatch.setattr("subprocess.run", runner)
    authorizer = GPUAuthorizer(nvat_command="nvattest")
    token = json.dumps(_evidence())

    assert authorizer.verify(token) is True
    assert authorizer.verify(token) is False

    command, _ = runner.calls[0]
    assert "attest" in command
    assert command[command.index("--gpu-evidence-source") + 1] == "file"
    assert command[command.index("--verifier") + 1] == "remote"
    assert command[command.index("--nras-url") + 1] == "https://nras.attestation.nvidia.com"
    assert "--relying-party-policy" in command
    assert runner.verify_evidence == _evidence()["evidences"]


def test_concurrent_verify_calls_are_parallel_and_replay_safe(monkeypatch):
    class TrackingRunner(FakeRunner):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.max_active = 0
            self.guard = threading.Lock()

        def __call__(self, command, **kwargs):
            with self.guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.02)
                return super().__call__(command, **kwargs)
            finally:
                with self.guard:
                    self.active -= 1

    runner = TrackingRunner()
    monkeypatch.setattr("subprocess.run", runner)
    authorizer = GPUAuthorizer(nvat_command="nvattest")
    token = json.dumps(_evidence())
    start = threading.Barrier(2)

    def verify():
        start.wait()
        return authorizer.verify(token)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [executor.submit(verify), executor.submit(verify)]
        results = [result.result() for result in results]

    assert sorted(results) == [False, True]
    assert runner.max_active == 2


def test_verify_rejects_failed_or_nonce_mismatched_appraisal(monkeypatch):
    runner = FakeRunner()
    monkeypatch.setattr("subprocess.run", runner)
    authorizer = GPUAuthorizer(nvat_command="nvattest")
    token = json.dumps(_evidence())

    runner.verify_result = subprocess.CompletedProcess([], 4, "{}")
    assert authorizer.verify(token) is False

    runner.verify_result = subprocess.CompletedProcess([], 0, json.dumps(_verification("b" * 64)))
    assert authorizer.verify(token) is False

    weak_policy_result = _verification()
    weak_policy_result["claims"][0]["secboot"] = False
    runner.verify_result = subprocess.CompletedProcess([], 0, json.dumps(weak_policy_result))
    assert authorizer.verify(token) is False


@pytest.mark.parametrize(
    "token",
    [
        "",
        "not-json",
        "{}",
        json.dumps({"result_code": 1, "evidences": []}),
        json.dumps({"result_code": 0, "evidences": []}),
        json.dumps({"result_code": 0, "evidences": [{"nonce": "short"}]}),
        json.dumps(
            {
                "result_code": 0,
                "evidences": [{"nonce": "a" * 64}, {"nonce": "b" * 64}],
            }
        ),
    ],
)
def test_verify_rejects_malformed_evidence_without_running_nvat(token, monkeypatch):
    runner = FakeRunner()
    monkeypatch.setattr("subprocess.run", runner)
    authorizer = GPUAuthorizer(nvat_command="nvattest")

    assert authorizer.verify(token) is False
    assert runner.calls == []


def test_constructor_validates_configuration(tmp_path):
    empty_policy = tmp_path / "empty.rego"
    empty_policy.write_text("")
    legacy_policy = tmp_path / "legacy.json"
    legacy_policy.write_text('{"version":"4.0"}')

    with pytest.raises(ValueError, match="verifier"):
        GPUAuthorizer(verifier="unsupported")
    with pytest.raises(ValueError, match="HTTPS"):
        GPUAuthorizer(nras_url="http://verifier.example")
    with pytest.raises(ValueError, match="must not be empty"):
        GPUAuthorizer(policy_file=str(empty_policy))
    with pytest.raises(ValueError, match="must be Rego"):
        GPUAuthorizer(policy_file=str(legacy_policy))


def test_legacy_verifier_url_is_normalized():
    authorizer = GPUAuthorizer("https://nras.attestation.nvidia.com/v4/attest/gpu")

    assert authorizer.nras_url == "https://nras.attestation.nvidia.com"

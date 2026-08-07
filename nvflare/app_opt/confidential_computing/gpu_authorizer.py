# Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import os
import re
import secrets
import subprocess
import tempfile

from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer, CCTokenGenerateError

from .utils import NonceHistory

GPU_NAMESPACE = "x-nv-gpu"
NVAT_SERVICE_KEY_ENV = "NV_ATTESTATION_SERVICE_KEY"
LEGACY_SERVICE_KEY_ENV = "NVIDIA_ATTESTATION_SERVICE_KEY"
DEFAULT_NRAS_URL = "https://nras.attestation.nvidia.com"

# NVAT applies this Rego policy to cryptographically verified claims. Keep the
# checks explicit so a successful CLI exit alone is never the authorization
# decision.
default_policy = """package policy
import future.keywords.every

default nv_match = false

nv_match {
  count(input) > 0
  every result in input {
    result["x-nvidia-device-type"] == "gpu"
    result["measres"] == "success"
    result["secboot"] == true
    result["dbgstat"] == "disabled"
    result["x-nvidia-gpu-arch-check"] == true
    result["x-nvidia-gpu-attestation-report-parsed"] == true
    result["x-nvidia-gpu-attestation-report-nonce-match"] == true
    result["x-nvidia-gpu-attestation-report-signature-verified"] == true
    result["x-nvidia-gpu-attestation-report-cert-chain"]["x-nvidia-cert-status"] == "valid"
    result["x-nvidia-gpu-attestation-report-cert-chain"]["x-nvidia-cert-ocsp-status"] == "good"
    result["x-nvidia-gpu-attestation-report-cert-chain-fwid-match"] == true
    result["x-nvidia-gpu-driver-rim-signature-verified"] == true
    result["x-nvidia-gpu-driver-rim-version-match"] == true
    result["x-nvidia-gpu-vbios-rim-signature-verified"] == true
    result["x-nvidia-gpu-vbios-rim-version-match"] == true
    result["x-nvidia-mismatch-measurement-records"] == null
  }
}
"""


def _default_policy_claim_matches(claim: dict) -> bool:
    """Evaluate the built-in Rego policy's security predicates independently."""
    certificate_chain = claim.get("x-nvidia-gpu-attestation-report-cert-chain")
    return (
        claim.get("x-nvidia-device-type") == "gpu"
        and claim.get("measres") == "success"
        and claim.get("secboot") is True
        and claim.get("dbgstat") == "disabled"
        and claim.get("x-nvidia-gpu-arch-check") is True
        and claim.get("x-nvidia-gpu-attestation-report-parsed") is True
        and claim.get("x-nvidia-gpu-attestation-report-nonce-match") is True
        and claim.get("x-nvidia-gpu-attestation-report-signature-verified") is True
        and isinstance(certificate_chain, dict)
        and certificate_chain.get("x-nvidia-cert-status") == "valid"
        and certificate_chain.get("x-nvidia-cert-ocsp-status") == "good"
        and claim.get("x-nvidia-gpu-attestation-report-cert-chain-fwid-match") is True
        and claim.get("x-nvidia-gpu-driver-rim-signature-verified") is True
        and claim.get("x-nvidia-gpu-driver-rim-version-match") is True
        and claim.get("x-nvidia-gpu-vbios-rim-signature-verified") is True
        and claim.get("x-nvidia-gpu-vbios-rim-version-match") is True
        and "x-nvidia-mismatch-measurement-records" in claim
        and claim["x-nvidia-mismatch-measurement-records"] is None
    )


def _parse_evidence_document(token: str, max_token_size: int) -> tuple[dict, str]:
    if not isinstance(token, str) or not token or len(token.encode("utf-8")) > max_token_size:
        raise ValueError("GPU evidence token is empty or exceeds the configured size limit")
    try:
        document = json.loads(token)
    except json.JSONDecodeError as e:
        raise ValueError("GPU evidence token is not valid JSON") from e
    if not isinstance(document, dict) or document.get("result_code") != 0:
        raise ValueError("GPU evidence collection did not report success")
    evidences = document.get("evidences")
    if not isinstance(evidences, list) or not evidences:
        raise ValueError("GPU evidence token contains no device evidence")

    nonce = None
    for evidence in evidences:
        if not isinstance(evidence, dict):
            raise ValueError("GPU evidence token contains an invalid evidence entry")
        evidence_nonce = evidence.get("nonce")
        if not isinstance(evidence_nonce, str) or not re.fullmatch(r"[0-9A-Fa-f]{64}", evidence_nonce):
            raise ValueError("GPU evidence token contains an invalid nonce")
        normalized_nonce = evidence_nonce.lower()
        if nonce is None:
            nonce = normalized_nonce
        elif nonce != normalized_nonce:
            raise ValueError("GPU evidence token contains inconsistent nonces")
    return document, nonce


class GPUAuthorizer(CCAuthorizer):
    """Generate and verify NVIDIA GPU evidence with the NVAT ``nvattest`` CLI."""

    def __init__(
        self,
        verifier_url=None,
        policy_file=None,
        max_nonce_history=1000,
        *,
        nvat_command="/opt/attestation/bin/nvattest",
        verifier="remote",
        nras_url=DEFAULT_NRAS_URL,
        service_key=None,
        cmd_timeout=180,
        max_token_size=16 * 1024 * 1024,
    ):
        """Initialize the NVAT-backed GPU authorizer.

        ``verifier_url`` is retained as a compatibility alias for deployments
        that configured the retired Python SDK authorizer.

        The ``policy_file`` marker checks below are only a lightweight
        configuration pre-check. NVAT remains authoritative: it compiles and
        evaluates the Rego policy during attestation, so a malformed policy or
        one that merely mentions ``nv_match`` in a comment fails closed then.
        """
        super().__init__()
        if not isinstance(nvat_command, str) or not nvat_command.strip():
            raise ValueError("nvat_command must be a non-empty string")
        if verifier not in ("local", "remote"):
            raise ValueError("verifier must be 'local' or 'remote'")
        if cmd_timeout <= 0:
            raise ValueError("cmd_timeout must be positive")
        if max_token_size <= 0:
            raise ValueError("max_token_size must be positive")

        if verifier_url:
            nras_url = verifier_url.removesuffix("/v4/attest/gpu")
        if verifier == "remote" and (not isinstance(nras_url, str) or not nras_url.startswith("https://")):
            raise ValueError("nras_url must be an HTTPS URL for remote verification")

        self.logger = logging.getLogger(self.__class__.__name__)
        self._can_generate = True
        self.my_nonce_history = NonceHistory(max_nonce_history)
        self.seen_nonce_history = NonceHistory(max_nonce_history)
        self.nvat_command = nvat_command
        self.verifier = verifier
        self.nras_url = nras_url.rstrip("/")
        self.cmd_timeout = cmd_timeout
        self.max_token_size = max_token_size

        if policy_file is None:
            self.relying_party_policy = default_policy
            self._uses_default_policy = True
        else:
            with open(policy_file, encoding="utf-8") as policy_stream:
                self.relying_party_policy = policy_stream.read()
            if not self.relying_party_policy.strip():
                raise ValueError("NVAT relying-party policy must not be empty")
            if "package " not in self.relying_party_policy or "nv_match" not in self.relying_party_policy:
                raise ValueError("NVAT relying-party policy must be Rego defining nv_match")
            self._uses_default_policy = False

        if service_key is None:
            service_key = os.environ.get(NVAT_SERVICE_KEY_ENV) or os.environ.get(LEGACY_SERVICE_KEY_ENV)
        self.service_key = service_key

    def _run(self, arguments: list[str], action: str) -> subprocess.CompletedProcess:
        environment = os.environ.copy()
        if self.service_key:
            environment[NVAT_SERVICE_KEY_ENV] = self.service_key
        try:
            return subprocess.run(
                [self.nvat_command, "--format", "json", "--log-level", "warn", *arguments],
                capture_output=True,
                text=True,
                timeout=self.cmd_timeout,
                start_new_session=True,
                env=environment,
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"NVAT executable not found: {self.nvat_command}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"NVAT {action} timed out after {self.cmd_timeout} seconds") from e
        except OSError as e:
            raise RuntimeError(f"NVAT {action} could not start: {e}") from e

    @staticmethod
    def _successful_result(result: subprocess.CompletedProcess, action: str) -> dict:
        if result.returncode != 0:
            raise RuntimeError(f"NVAT {action} failed with exit status {result.returncode}")
        try:
            document = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"NVAT {action} returned invalid JSON") from e
        if not isinstance(document, dict) or document.get("result_code") != 0:
            raise RuntimeError(f"NVAT {action} did not report success")
        return document

    def generate(self) -> str:
        nonce = secrets.token_hex(32)
        try:
            result = self._run(
                ["collect-evidence", "--device", "gpu", "--nonce", nonce],
                "GPU evidence collection",
            )
            document = self._successful_result(result, "GPU evidence collection")
            token = json.dumps(document, separators=(",", ":"), sort_keys=True)
            _, evidence_nonce = _parse_evidence_document(token, self.max_token_size)
            if evidence_nonce != nonce:
                raise RuntimeError("NVAT GPU evidence nonce does not match the generated challenge")
            self.my_nonce_history.add(nonce)
            self._can_generate = True
            return token
        except (RuntimeError, ValueError) as e:
            self._can_generate = False
            raise CCTokenGenerateError(f"Failed to generate NVAT GPU evidence: {e}") from e

    def verify(self, token: str) -> bool:
        try:
            document, nonce = _parse_evidence_document(token, self.max_token_size)
            with tempfile.TemporaryDirectory(prefix="nvflare-nvat-verify-") as work_dir:
                evidence_path = os.path.join(work_dir, "gpu-evidence.json")
                policy_path = os.path.join(work_dir, "relying-party-policy.rego")
                with open(evidence_path, "w", encoding="utf-8") as evidence_stream:
                    # NVAT collect-evidence returns a result wrapper, while the
                    # 1.2.2 JSON file evidence-source API consumes the evidence
                    # array itself.
                    json.dump(document["evidences"], evidence_stream, separators=(",", ":"), sort_keys=True)
                with open(policy_path, "w", encoding="utf-8") as policy_stream:
                    policy_stream.write(self.relying_party_policy)

                command = [
                    "attest",
                    "--device",
                    "gpu",
                    "--verifier",
                    self.verifier,
                    "--nonce",
                    nonce,
                    "--gpu-evidence-source",
                    "file",
                    "--gpu-evidence-file",
                    evidence_path,
                    "--relying-party-policy",
                    policy_path,
                ]
                if self.verifier == "remote":
                    command.extend(["--nras-url", self.nras_url])
                result = self._run(command, "GPU evidence verification")
                verified = self._successful_result(result, "GPU evidence verification")
                claims = verified.get("claims")
                if not isinstance(claims, list) or not claims:
                    raise RuntimeError("NVAT verification returned no GPU claims")
                if any(
                    not isinstance(claim, dict) or str(claim.get("eat_nonce", "")).lower() != nonce for claim in claims
                ):
                    raise RuntimeError("NVAT verified claims do not match the evidence nonce")
                # NVAT 1.2.2 documents that nv_match=false returns
                # NVAT_RP_POLICY_MISMATCH. Re-evaluate every predicate in the
                # built-in policy as defense in depth, so authorization never
                # depends only on the CLI exit/result-code contract.
                # https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/1.2.2/sdk-cli/command-reference.html
                if self._uses_default_policy and not all(_default_policy_claim_matches(claim) for claim in claims):
                    raise RuntimeError("NVAT verified claims do not satisfy the built-in relying-party policy")
                if not self.seen_nonce_history.add(nonce):
                    self.logger.warning("Rejected replayed NVAT GPU evidence")
                    return False
            return True
        except (OSError, RuntimeError, ValueError) as e:
            self.logger.warning("NVAT GPU evidence verification failed: %s", e)
            return False

    def get_namespace(self) -> str:
        return GPU_NAMESPACE

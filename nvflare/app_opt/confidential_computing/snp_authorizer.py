# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import binascii
import logging
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager

from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer, CCTokenGenerateError

from .utils import NonceHistory

SNP_NAMESPACE = "x-snp"
REPORT_PATH = "report.bin"
REQUEST_PATH = "request.bin"

AMD_ARK = "ark.pem"
AMD_ASK = "ask.pem"
AMD_VCEK = "vcek.pem"

# These fields are stable parts of the AMD SEV-SNP attestation report ABI.  Reading
# them directly avoids depending on the human-readable output of a particular
# snpguest release.
REPORT_DATA_OFFSET = 80
REPORT_DATA_SIZE = 64
REPORTED_TCB_OFFSET = 384
REPORTED_TCB_SIZE = 8
CHIP_ID_OFFSET = 416
CHIP_ID_SIZE = 64
MIN_REPORT_SIZE = CHIP_ID_OFFSET + CHIP_ID_SIZE


def parse_chip_id(report_text: str) -> str:
    """Parse a chip ID from legacy ``snpguest display report`` output."""
    match = re.search(
        r"Chip ID:\s*((?:[0-9A-Fa-f]{2}\s+){15}[0-9A-Fa-f]{2}" r"(?:\s*\n\s*(?:[0-9A-Fa-f]{2}\s+){15}[0-9A-Fa-f]{2})*)",
        report_text,
        re.MULTILINE,
    )
    return "" if not match else "".join(match.group(1).split()).lower()


def parse_reported_tcb(report_text: str) -> dict:
    """Parse reported TCB fields from legacy ``snpguest`` text output."""
    match = re.search(
        r"Reported TCB:\s*"
        r"TCB Version:\s*"
        r"Microcode:\s*(\d+)\s*"
        r"SNP:\s*(\d+)\s*"
        r"TEE:\s*(\d+)\s*"
        r"Boot Loader:\s*(\d+)\s*"
        r"FMC:\s*(\w+)",
        report_text,
        re.MULTILINE,
    )
    if not match:
        return {}
    microcode, snp, tee, boot_loader, fmc = match.groups()
    return {
        "Microcode": int(microcode),
        "SNP": int(snp),
        "TEE": int(tee),
        "Boot Loader": int(boot_loader),
        "FMC": None if fmc == "None" else fmc,
    }


def _read_report_field(report: bytes, offset: int, size: int, field_name: str) -> bytes:
    if len(report) < offset + size:
        raise ValueError(f"SNP report is too short to contain {field_name}")
    return report[offset : offset + size]


@contextmanager
def _file_lock(path: str, timeout: float):
    """Take an inter-process lock without adding a runtime Python dependency."""
    import fcntl  # SNP attestation is Linux-only; keep the import local for portability.

    lock_file = open(path, "a+")
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for lock: {path}")
                time.sleep(0.1)
        yield
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


class SNPAuthorizer(CCAuthorizer):
    """Generate and verify AMD SEV-SNP attestation reports with ``snpguest``."""

    def __init__(
        self,
        max_nonce_history=1000,
        amd_certs_dir="/opt/certs",
        snpguest_binary="snpguest",
        cpu_model="milan",
        max_retries=5,
        retry_interval=10,
        cmd_timeout=60,
        lock_timeout=60,
    ):
        """Initialize the SNP authorizer.

        ``amd_certs_dir`` is a persistent cache for AMD ARK/ASK and per-chip
        VCEK certificates. Request and report files are created in isolated
        temporary directories and removed after each operation.
        """
        super().__init__()
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if cmd_timeout <= 0 or lock_timeout <= 0:
            raise ValueError("cmd_timeout and lock_timeout must be positive")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.my_nonce_history = NonceHistory(max_nonce_history)
        self.seen_nonce_history = NonceHistory(max_nonce_history)
        self.amd_certs_dir = os.path.abspath(amd_certs_dir)
        self.snpguest_binary = snpguest_binary
        self.cpu_model = cpu_model
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.cmd_timeout = cmd_timeout
        self.lock_timeout = lock_timeout

    def _run_with_retry(self, cmd: list[str], action_name: str) -> subprocess.CompletedProcess:
        last_error = "unknown error"
        for attempt in range(1, self.max_retries + 1):
            self.logger.info("[%s] Attempt %d/%d", action_name, attempt, self.max_retries)
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.cmd_timeout)
            except FileNotFoundError as e:
                raise RuntimeError(f"[{action_name}] executable not found: {cmd[0]}") from e
            except subprocess.TimeoutExpired:
                last_error = f"command timed out after {self.cmd_timeout} seconds"
                self.logger.warning("[%s] %s", action_name, last_error)
            except OSError as e:
                last_error = str(e)
                self.logger.warning("[%s] command failed to start: %s", action_name, e)
            else:
                if result.returncode == 0:
                    return result
                last_error = result.stderr.strip() or f"exit status {result.returncode}"
                self.logger.warning("[%s] Failed: %s", action_name, last_error)

            if attempt < self.max_retries:
                time.sleep(min(self.retry_interval * 2 ** (attempt - 1), 60))
        raise RuntimeError(f"[{action_name}] failed after {self.max_retries} attempts: {last_error}")

    def _run_once(self, cmd: list[str], action_name: str) -> subprocess.CompletedProcess:
        """Run a deterministic local operation exactly once."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.cmd_timeout)
        except FileNotFoundError as e:
            raise RuntimeError(f"[{action_name}] executable not found: {cmd[0]}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"[{action_name}] command timed out after {self.cmd_timeout} seconds") from e
        except OSError as e:
            raise RuntimeError(f"[{action_name}] command failed to start: {e}") from e
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit status {result.returncode}"
            raise RuntimeError(f"[{action_name}] failed: {detail}")
        return result

    def generate(self) -> str:
        nonce = secrets.token_bytes(REPORT_DATA_SIZE)
        try:
            with tempfile.TemporaryDirectory(prefix="nvflare-snp-generate-") as work_dir:
                request_path = os.path.join(work_dir, "request.bin")
                report_path = os.path.join(work_dir, "report.bin")
                with open(request_path, "wb") as request_file:
                    request_file.write(nonce)

                self._run_with_retry([self.snpguest_binary, "report", report_path, request_path], "generate_report")
                with open(report_path, "rb") as report_file:
                    report = report_file.read()
                if len(report) < MIN_REPORT_SIZE:
                    raise RuntimeError("snpguest generated an incomplete SNP report")
        except (OSError, RuntimeError) as e:
            raise CCTokenGenerateError(f"Failed to generate an SNP attestation report: {e}") from e

        self.my_nonce_history.add(nonce.hex())
        return base64.b64encode(report).decode("ascii")

    def verify(self, token: str) -> bool:
        try:
            if not isinstance(token, (str, bytes)) or not token:
                return False
            report = base64.b64decode(token, validate=True)
            if len(report) < MIN_REPORT_SIZE:
                return False

            os.makedirs(self.amd_certs_dir, mode=0o700, exist_ok=True)
            self._ensure_amd_ca_certs()
            with tempfile.TemporaryDirectory(prefix="nvflare-snp-verify-") as work_dir:
                report_path = os.path.join(work_dir, "report.bin")
                with open(report_path, "wb") as report_file:
                    report_file.write(report)

                vcek_cache_key = self._parse_report(report_path)
                vcek_path = self._ensure_amd_vcek(vcek_cache_key, report_path)
                for filename in (AMD_ARK, AMD_ASK):
                    shutil.copy2(os.path.join(self.amd_certs_dir, filename), os.path.join(work_dir, filename))
                shutil.copy2(vcek_path, os.path.join(work_dir, AMD_VCEK))

                # Signature verification is local and deterministic. Retrying a
                # bad report only amplifies an attacker's resource consumption;
                # bounded retries are reserved for the AMD certificate fetches.
                self._run_once(
                    [self.snpguest_binary, "verify", "attestation", work_dir, report_path],
                    "verify_attestation",
                )

            report_data = _read_report_field(report, REPORT_DATA_OFFSET, REPORT_DATA_SIZE, "report data")
            if not any(report_data):
                self.logger.warning("Rejected an SNP attestation report without a freshness nonce")
                return False
            nonce = report_data.hex()
            if not self.seen_nonce_history.add(nonce):
                self.logger.warning("Rejected a replayed SNP attestation report")
                return False
            self.logger.info("SNP attestation and nonce checks passed")
            return True
        except (binascii.Error, OSError, RuntimeError, TimeoutError, ValueError) as e:
            self.logger.warning("SNP token verification failed: %s", e)
            return False

    def _ensure_amd_ca_certs(self) -> None:
        """Fetch the AMD CA certificates once, with cross-process serialization."""
        os.makedirs(self.amd_certs_dir, mode=0o700, exist_ok=True)
        lock_path = os.path.join(self.amd_certs_dir, ".ca.lock")
        with _file_lock(lock_path, self.lock_timeout):
            ask_path = os.path.join(self.amd_certs_dir, AMD_ASK)
            ark_path = os.path.join(self.amd_certs_dir, AMD_ARK)
            if os.path.isfile(ark_path) and os.path.isfile(ask_path):
                return
            self._run_with_retry(
                [self.snpguest_binary, "fetch", "ca", "pem", self.amd_certs_dir, self.cpu_model],
                "fetch_ca_certs",
            )
            if not (os.path.isfile(ark_path) and os.path.isfile(ask_path)):
                raise RuntimeError("snpguest did not create the expected AMD ARK and ASK certificates")

    def _ensure_amd_vcek(self, vcek_cache_key: str, report_bin_file: str) -> str:
        """Return a cached VCEK, fetching it safely when it is not present."""
        cache_path = os.path.join(self.amd_certs_dir, f"vcek-{vcek_cache_key}.pem")
        lock_path = os.path.join(self.amd_certs_dir, ".vcek.lock")
        with _file_lock(lock_path, self.lock_timeout):
            if not os.path.isfile(cache_path):
                with tempfile.TemporaryDirectory(prefix="nvflare-snp-vcek-") as cert_dir:
                    self._run_with_retry(
                        [self.snpguest_binary, "fetch", "vcek", "pem", cert_dir, report_bin_file],
                        "fetch_vcek",
                    )
                    fetched_vcek = os.path.join(cert_dir, AMD_VCEK)
                    if not os.path.isfile(fetched_vcek):
                        raise RuntimeError("snpguest did not create the expected VCEK certificate")
                    temporary_cache = f"{cache_path}.{os.getpid()}.tmp"
                    shutil.copy2(fetched_vcek, temporary_cache)
                    os.replace(temporary_cache, cache_path)
        return cache_path

    @staticmethod
    def _parse_report(report_bin_file: str) -> str:
        """Build a bounded VCEK cache key from the report's chip ID and reported TCB."""
        with open(report_bin_file, "rb") as report_file:
            report = report_file.read()
        chip_id = _read_report_field(report, CHIP_ID_OFFSET, CHIP_ID_SIZE, "chip ID")
        reported_tcb = _read_report_field(report, REPORTED_TCB_OFFSET, REPORTED_TCB_SIZE, "reported TCB")
        return f"{chip_id.hex()}-{reported_tcb.hex()}"

    def get_namespace(self) -> str:
        return SNP_NAMESPACE

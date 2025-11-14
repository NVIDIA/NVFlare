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
import logging
import os
import random
import re
import shutil
import subprocess
import time
import uuid

from filelock import FileLock

from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer

from .utils import NonceHistory

SNP_NAMESPACE = "x-snp"
REPORT_PATH = "report.bin"
REQUEST_PATH = "request.bin"

AMD_ARK = "ark.pem"
AMD_ASK = "ask.pem"
AMD_VCEK = "vcek.pem"


def parse_chip_id(report_text: str) -> str:
    # Find the block starting with "Chip ID:" followed by multiple lines of hex bytes
    match = re.search(
        r"Chip ID:\s*((?:[0-9A-Fa-f]{2}\s+){15}[0-9A-Fa-f]{2}(?:\s*\n\s*(?:[0-9A-Fa-f]{2}\s+){15}[0-9A-Fa-f]{2})*)",
        report_text,
        re.MULTILINE,
    )
    if not match:
        return ""

    # Extract all hex bytes and remove spaces/newlines
    hex_block = match.group(1)
    # Remove all whitespace characters and convert to lowercase
    chip_id = "".join(hex_block.split()).lower()

    return chip_id


def parse_reported_tcb(report_text: str) -> dict:
    # Match the entire Reported TCB block after the line "Reported TCB:"
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

    # Parse FMC which can be 'None' or something else
    microcode, snp, tee, boot_loader, fmc = match.groups()

    return {
        "Microcode": int(microcode),
        "SNP": int(snp),
        "TEE": int(tee),
        "Boot Loader": int(boot_loader),
        "FMC": None if fmc == "None" else fmc,
    }


class SNPAuthorizer(CCAuthorizer):
    """AMD SEV-SNP Authorizer"""

    def __init__(
        self,
        max_nonce_history=1000,
        amd_certs_dir="/opt/certs",
        snpguest_binary="snpguest",
        cpu_model="milan",
        max_retries=3,
        retry_interval=5,
        cmd_timeout=60,
    ):
        """
        Initialize the SNPAuthorizer instance.

        Args:
            max_nonce_history (int, optional): Maximum number of nonces to keep in history for replay protection.
                Defaults to 1000.
            amd_certs_dir (str, optional): Directory path where AMD certificates are stored.
                Defaults to "/opt/certs".
            snpguest_binary (str, optional): Path to the `snpguest` binary used for generating and verifying reports.
                Defaults to "/host/bin/snpguest".
            cpu_model (str, optional): CPU model identifier used when fetching certificates.
                Defaults to "milan".
            max_retries (int): Max number of retries on transient failures.
            retry_interval (int): Wait time (seconds) between retries.
            cmd_timeout (int): SNPGuest command timeout.
        """
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.my_nonce_history = NonceHistory(max_nonce_history)
        self.seen_nonce_history = NonceHistory(max_nonce_history)
        self.amd_certs_dir = amd_certs_dir
        self.snpguest_binary = snpguest_binary
        self.cpu_model = cpu_model
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.cmd_timeout = cmd_timeout

    def _run_with_retry(self, cmd: list[str], action_name: str) -> subprocess.CompletedProcess:
        for attempt in range(1, self.max_retries + 1):
            self.logger.info(f"[{action_name}] Attempt {attempt}/{self.max_retries}: running {cmd}")
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=self.cmd_timeout)
                if result.returncode == 0:
                    return result
                else:
                    self.logger.warning(
                        f"[{action_name}] Failed with return code {result.returncode}. "
                        f"stderr: {result.stderr.decode().strip()}"
                    )
            except subprocess.TimeoutExpired:
                self.logger.warning(f"[{action_name}] Command timed out.")

            if attempt < self.max_retries:
                time.sleep(min(self.retry_interval * 2 ** (attempt - 1), 60))  # Exponential backoff
        raise RuntimeError(f"[{action_name}] Failed after {self.max_retries} attempts.")

    def generate(self):
        nonce = bytearray([random.randint(0, 255) for _ in range(64)])
        with open(REQUEST_PATH, "wb") as request_file:
            request_file.write(nonce)

        cmd = [self.snpguest_binary, "report", REPORT_PATH, REQUEST_PATH]
        self._run_with_retry(cmd, "generate_report")

        with open(REPORT_PATH, "rb") as report_file:
            token = base64.b64encode(report_file.read())

        self.my_nonce_history.add(nonce)
        return token

    def verify(self, token):
        tmp_bin_file = uuid.uuid4().hex
        try:
            self._ensure_amd_ca_certs()
            report_bin = base64.b64decode(token)
            with open(tmp_bin_file, "wb") as report_file:
                report_file.write(report_bin)

            vcek_cache_key = self._parse_report(tmp_bin_file)
            self._ensure_amd_vcek(vcek_cache_key, tmp_bin_file)

            # Verify attestation
            cmd = [self.snpguest_binary, "verify", "attestation", self.amd_certs_dir, tmp_bin_file]
            result = self._run_with_retry(cmd, "verify_attestation")

            if result.returncode == 0:
                self.logger.info("Attestation passed")
                if self._check_nonce(tmp_bin_file):
                    self.logger.info("Check nonce passed")
                    return True
                else:
                    self.logger.info("Check nonce failed")
                    return False
            else:
                self.logger.warning("Attestation verification failed.")
                return False

        except Exception as e:
            self.logger.error(f"Token verification failed: {e}")
            return False
        finally:
            if os.path.exists(tmp_bin_file):
                os.remove(tmp_bin_file)

    def _ensure_amd_ca_certs(self):
        """Ensures AMD CA certs are inside the amd_certs_dir."""
        ask_path = os.path.join(self.amd_certs_dir, AMD_ASK)
        ark_path = os.path.join(self.amd_certs_dir, AMD_ARK)
        if not (os.path.exists(ark_path) and os.path.exists(ask_path)):
            self.logger.info("AMD CA certs not found. Fetching...")
            cmd = [self.snpguest_binary, "fetch", "ca", "pem", self.amd_certs_dir, self.cpu_model]
            self._run_with_retry(cmd, "fetch_ca_certs")
        else:
            self.logger.info("AMD CA certs already exist.")

    def _ensure_amd_vcek(self, vcek_cache_key, report_bin_file, timeout=60):
        """Ensures AMD VCEK is inside the amd_certs_dir."""
        cache_path = os.path.join(self.amd_certs_dir, vcek_cache_key)
        vcek_file = os.path.join(self.amd_certs_dir, AMD_VCEK)
        lock_file = cache_path + ".lock"
        with FileLock(lock_file, timeout=timeout):
            if not os.path.exists(cache_path):
                self.logger.info("AMD VCEK not cached. Fetching and caching...")
                cmd = [self.snpguest_binary, "fetch", "vcek", "pem", self.amd_certs_dir, report_bin_file]
                self._run_with_retry(cmd, "fetch_vcek")
                if not os.path.exists(vcek_file):
                    raise RuntimeError(f"VCEK file not generated at expected path: {vcek_file}")
                # Rename vcek.pem to the cache file name
                shutil.move(vcek_file, cache_path)
            else:
                self.logger.info("Using cached AMD VCEK")
        shutil.copy(cache_path, vcek_file)

    def _parse_report(self, report_bin_file):
        """Parses the Reported TCB and Chip ID info.

        This method is used to generate a unique id to cache VCEK.
        Because AMD KDS has rate limitation, we should avoid keep polling.
        """
        cmd = [self.snpguest_binary, "display", "report", report_bin_file]
        cp = subprocess.run(cmd, capture_output=True)
        if cp.returncode != 0:
            self.logger.error("Can't display SNP report")
            raise RuntimeError("Can't display SNP report")
        output_string = cp.stdout
        report_text = output_string.decode("utf-8")
        chip_id = parse_chip_id(report_text)
        reported_tcb = parse_reported_tcb(report_text)
        if not reported_tcb:
            raise RuntimeError("Failed to parse Reported TCB from report")
        cache_key = f"{chip_id}-{reported_tcb['Microcode']}-{reported_tcb['SNP']}-{reported_tcb['TEE']}-{reported_tcb['Boot Loader']}"
        return cache_key

    def _check_nonce(self, report_bin_file):
        """Parses nonce from the Report Data section and checks if it is fresh."""
        cmd = [self.snpguest_binary, "display", "report", report_bin_file]
        cp = subprocess.run(cmd, capture_output=True)
        if cp.returncode != 0:
            return False
        output_string = cp.stdout
        lines = output_string.decode("utf-8").split("\n")
        report_data_string = ""
        for i in range(len(lines)):
            if lines[i] == "Report Data:":
                report_data_string = " ".join(lines[i + 1 : i + 6]).replace(" ", "")
                break
        return self.seen_nonce_history.add(report_data_string)

    def get_namespace(self) -> str:
        return SNP_NAMESPACE

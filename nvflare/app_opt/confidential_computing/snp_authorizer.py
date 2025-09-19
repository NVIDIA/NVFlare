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
import subprocess
import uuid

from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer

from .utils import NonceHistory

SNP_NAMESPACE = "x-snp"

REPORT_PATH = "report.bin"
REQUEST_PATH = "request.bin"


class SNPAuthorizer(CCAuthorizer):
    """AMD SEV-SNP Authorizer"""

    def __init__(
        self,
        max_nonce_history=1000,
        amd_certs_dir="/opt/certs",
        snpguest_binary="snpguest",
        cpu_model="milan",
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

        """

        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.my_nonce_history = NonceHistory(max_nonce_history)
        self.seen_nonce_history = NonceHistory(max_nonce_history)
        self.amd_certs_dir = amd_certs_dir
        self.snpguest_binary = snpguest_binary
        self.cpu_model = cpu_model

    def _ensure_amd_ca_certs(self):
        ask_path = os.path.join(self.amd_certs_dir, "ask.pem")
        ark_path = os.path.join(self.amd_certs_dir, "ark.pem")
        if not (os.path.exists(ark_path) and os.path.exists(ask_path)):
            self.logger.info("AMD CA certs not found. Fetching...")
            cmd = [self.snpguest_binary, "fetch", "ca", "pem", self.amd_certs_dir, self.cpu_model]
            subprocess.run(cmd, capture_output=True)
        else:
            self.logger.info("AMD CA certs already exist.")

    def generate(self):
        nonce = bytearray([random.randint(0, 255) for _ in range(64)])
        with open(REQUEST_PATH, "wb") as request_file:
            request_file.write(nonce)

        cmd = [self.snpguest_binary, "report", REPORT_PATH, REQUEST_PATH]
        subprocess.run(cmd, capture_output=True)

        with open(REPORT_PATH, "rb") as report_file:
            token = base64.b64encode(report_file.read())

        self.my_nonce_history.add(nonce)
        return token

    def verify(self, token):
        try:
            self._ensure_amd_ca_certs()
            report_bin = base64.b64decode(token)
            tmp_bin_file = uuid.uuid4().hex
            with open(tmp_bin_file, "wb") as report_file:
                report_file.write(report_bin)
            cmd = [self.snpguest_binary, "fetch", "vcek", "pem", self.amd_certs_dir, tmp_bin_file]
            cp = subprocess.run(cmd, capture_output=True)
            if cp.returncode != 0:
                return False
            cmd = [self.snpguest_binary, "verify", "attestation", self.amd_certs_dir, tmp_bin_file]
            cp = subprocess.run(cmd, capture_output=True)
            if cp.returncode == 0:
                return self._check_nonce(tmp_bin_file)
            else:
                return False
        except Exception as e:
            self.logger.info(f"Token verification failed {e=}")
            return False
        finally:
            if os.path.exists(tmp_bin_file):
                os.remove(tmp_bin_file)

    def _check_nonce(self, tmp_bin_file):
        cmd = [self.snpguest_binary, "display", "report", tmp_bin_file]
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

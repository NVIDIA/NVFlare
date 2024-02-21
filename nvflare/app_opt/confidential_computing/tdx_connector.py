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
import logging
import os
import shlex
import subprocess
from typing import Dict

# TDX_CLI_COMMAND = "./trustauthority-cli"
TDX_CLI_CONFIG = "config.json"
TOKEN_FILE = "token.txt"
VERIFY_FILE = "verify.txt"
ERROR_FILE = "error.txt"


class TDXConnector:
    def __init__(self, tdx_cli_command: str, config_dir: str) -> None:
        super().__init__()
        self.tdx_cli_command = tdx_cli_command
        self.config_dir = config_dir

        self.config_file = os.path.join(self.config_dir, TDX_CLI_CONFIG)

    def get_token(self):
        token_file = os.path.join(self.config_dir, TOKEN_FILE)
        out = open(token_file, "w")
        error_file = os.path.join(self.config_dir, ERROR_FILE)
        err_out = open(error_file, "w")

        command = "sudo " + self.tdx_cli_command + " -c " + self.config_file + " token --no-eventlog "
        process = subprocess.Popen(shlex.split(command, True), preexec_fn=os.setsid, stdout=out, stderr=err_out)
        process.wait()

        with open(token_file, "r") as f:
            token = f.readline()
        with open(error_file, "r") as f:
            if 'Error:' in f.read():
                error = True
            else:
                error = False

        return token, error

    def verify_token(self, token: str):
        out = open(os.path.join(self.config_dir, VERIFY_FILE), "w")
        error_file = os.path.join(self.config_dir, ERROR_FILE)
        err_out = open(error_file, "w")

        command = self.tdx_cli_command + " verify --config " + self.config_file + " --token " + token
        process = subprocess.Popen(shlex.split(command, True), preexec_fn=os.setsid, stdout=out, stderr=err_out)
        process.wait()

        # with open(VERIFY_FILE, "r") as f:
        #     result = f.readline()
        with open(error_file, "r") as f:
            if 'Error:' in f.read():
                return False

        return True


class TDXCCHelper:

    def __init__(self, site_name: str, tdx_cli_command: str, config_dir: str) -> None:
        super().__init__()
        self.site_name = site_name
        # self.tdx_cli_command = tdx_cli_command
        # self.config_dir = config_dir
        self.token = None

        self.tdx_connector = TDXConnector(tdx_cli_command, config_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

    def prepare(self) -> bool:
        self.token, error = self.tdx_connector.get_token()
        self.logger.info(f"site: {self.site_name} got the token: {self.token}")
        return not error

    def get_token(self):
        return self.token

    def validate_participants(self, participants: Dict[str, str]) -> Dict[str, bool]:
        result = {}
        if not participants:
            return result
        for k, v in participants.items():
            if self.tdx_connector.verify_token(v):
                result[k] = True
        self.logger.info(f"CC - results from validating participants' tokens: {result}")
        return result


if __name__ == "__main__":
    tdx_connector = TDXConnector()
    token, error = tdx_connector.get_token()
    print("--- Acquire the token ---")
    print(token)

    result = tdx_connector.verify_token(token)
    print("---- Verify the token ---")
    print(result)

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

import os
import subprocess

from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer

TDX_NAMESPACE = "tdx_"
TDX_CLI_CONFIG = "config.json"
TOKEN_FILE = "token.txt"
VERIFY_FILE = "verify.txt"
ERROR_FILE = "error.txt"


class TDXAuthorizer(CCAuthorizer):
    def __init__(self, tdx_cli_command: str, config_dir: str) -> None:
        super().__init__()
        self.tdx_cli_command = tdx_cli_command
        self.config_dir = config_dir

        self.config_file = os.path.join(self.config_dir, TDX_CLI_CONFIG)

    def generate(self) -> str:
        token_file = os.path.join(self.config_dir, TOKEN_FILE)
        out = open(token_file, "w")
        error_file = os.path.join(self.config_dir, ERROR_FILE)
        err_out = open(error_file, "w")

        command = ["sudo", self.tdx_cli_command, "-c", self.config_file, "token", "--no-eventlog"]
        subprocess.run(command, preexec_fn=os.setsid, stdout=out, stderr=err_out)

        if not os.path.exists(error_file) or not os.path.exists(token_file):
            return ""

        try:
            with open(error_file, "r") as e_f:
                if "Error:" in e_f.read():
                    return ""
                else:
                    with open(token_file, "r") as t_f:
                        token = t_f.readline()
                    return token
        except:
            return ""

    def verify(self, token: str) -> bool:
        out = open(os.path.join(self.config_dir, VERIFY_FILE), "w")
        error_file = os.path.join(self.config_dir, ERROR_FILE)
        err_out = open(error_file, "w")

        command = [self.tdx_cli_command, "verify", "--config", self.config_file, "--token", token]
        subprocess.run(command, preexec_fn=os.setsid, stdout=out, stderr=err_out)

        if not os.path.exists(error_file):
            return False

        try:
            with open(error_file, "r") as f:
                if "Error:" in f.read():
                    return False
        except:
            return False

        return True

    def get_namespace(self) -> str:
        return TDX_NAMESPACE

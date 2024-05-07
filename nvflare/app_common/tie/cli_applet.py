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
import shlex
import subprocess

from nvflare.security.logging import secure_format_exception

from .applet import Applet
from .defs import Constant


class CLIApplet(Applet):

    def __init__(self):
        Applet.__init__(self)
        self._process = None
        self._start_error = False

    def start(self, ctx: dict):
        cli_cmd = ctx.get(Constant.APP_CTX_KEY_CLI_CMD)
        if not cli_cmd:
            raise RuntimeError(f"missing {Constant.APP_CTX_KEY_CLI_CMD} from app context")

        cli_cwd = ctx.get(Constant.APP_CTX_KEY_CLI_CWD)
        cli_env = ctx.get(Constant.APP_CTX_KEY_CLI_ENV)
        env = os.environ.copy()
        if cli_env:
            if not isinstance(cli_env, dict):
                raise RuntimeError(f"expect {Constant.APP_CTX_KEY_CLI_ENV} to be dict but got {type(cli_env)}")
            env.update(cli_env)

        command_seq = shlex.split(cli_cmd)

        try:
            self._process = subprocess.Popen(
                command_seq,
                stderr=subprocess.STDOUT,
                cwd=cli_cwd,
                env=env,
            )
        except Exception as ex:
            self.logger.error(f"exception starting applet: {secure_format_exception(ex)}")
            self._start_error = True

    def stop(self):
        p = self._process
        self._process = None
        if p:
            p.kill()

    def is_stopped(self) -> (bool, int):
        if self._start_error:
            return True, Constant.EXIT_CODE_CANT_START

        if self._process:
            return_code = self._process.poll()
            if return_code is None:
                return False, 0
            else:
                return True, return_code
        else:
            return True, 0

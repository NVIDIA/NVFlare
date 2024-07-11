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
import shlex
import subprocess

from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.app_common.tie.applet import Applet
from nvflare.app_common.tie.cli_applet import CLIApplet
from nvflare.app_common.tie.defs import Constant as TieConstant
from nvflare.app_opt.flower.defs import Constant
from nvflare.app_opt.flower.grpc_util import create_channel
from nvflare.fuel.f3.drivers.net_utils import get_open_tcp_port
from nvflare.security.logging import secure_format_exception


class FlowerClientApplet(CLIApplet):
    def __init__(
        self,
        client_app: str,
    ):
        CLIApplet.__init__(self)
        self.client_app = client_app

    def get_command(self, ctx: dict) -> (str, str, dict):
        addr = ctx.get(Constant.APP_CTX_SERVER_ADDR)
        fl_ctx = ctx.get(Constant.APP_CTX_FL_CONTEXT)
        if not isinstance(fl_ctx, FLContext):
            self.logger.error(f"expect APP_CTX_FL_CONTEXT to be FLContext but got {type(fl_ctx)}")
            raise RuntimeError("invalid FLContext")

        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        if not isinstance(ws, Workspace):
            self.logger.error(f"expect workspace to be Workspace but got {type(ws)}")
            raise RuntimeError("invalid workspace")

        custom_dir = ws.get_app_custom_dir(fl_ctx.get_job_id())
        cmd = f"flower-client-app --insecure --grpc-adapter --superlink {addr} --dir {custom_dir} {self.client_app}"
        self.logger.info(f"starting flower client app: {cmd}")
        return cmd, None, None


class FlowerServerApplet(Applet):
    def __init__(self, server_app: str, database: str, superlink_ready_timeout: float):
        Applet.__init__(self)
        self._app_process = None
        self._superlink_process = None
        self.server_app = server_app
        self.database = database
        self.superlink_ready_timeout = superlink_ready_timeout
        self._start_error = False

    def _start_process(self, name: str, cmd: str):
        self.logger.info(f"starting {name}: {cmd}")
        command_seq = shlex.split(cmd)
        try:
            return subprocess.Popen(
                command_seq,
                stderr=subprocess.STDOUT,
            )
        except Exception as ex:
            self.logger.error(f"exception starting applet: {secure_format_exception(ex)}")
            self._start_error = True
            return None

    def start(self, ctx: dict):
        # try to start superlink first
        driver_port = get_open_tcp_port(resources={})
        if not driver_port:
            raise RuntimeError("failed to get a port for Flower driver")
        driver_addr = f"127.0.0.1:{driver_port}"

        server_addr = ctx.get(Constant.APP_CTX_SERVER_ADDR)
        fl_ctx = ctx.get(Constant.APP_CTX_FL_CONTEXT)
        if not isinstance(fl_ctx, FLContext):
            self.logger.error(f"expect APP_CTX_FL_CONTEXT to be FLContext but got {type(fl_ctx)}")
            raise RuntimeError("invalid FLContext")

        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        if not isinstance(ws, Workspace):
            self.logger.error(f"expect workspace to be Workspace but got {type(ws)}")
            raise RuntimeError("invalid workspace")

        custom_dir = ws.get_app_custom_dir(fl_ctx.get_job_id())

        db_arg = ""
        if self.database:
            db_arg = f"--database {self.database}"

        superlink_cmd = (
            f"flower-superlink --insecure {db_arg} "
            f"--fleet-api-address {server_addr} --fleet-api-type grpc-adapter "
            f"--driver-api-address {driver_addr}"
        )

        self._superlink_process = self._start_process("superlink", superlink_cmd)
        if not self._superlink_process:
            raise RuntimeError("cannot start superlink process")

        # wait until superlink's port is ready before starting server app
        # note: the server app will connect to driver_addr, not server_addr
        create_channel(
            server_addr=driver_addr,
            grpc_options=None,
            ready_timeout=self.superlink_ready_timeout,
            test_only=True,
        )

        # start the server app
        app_cmd = f"flower-server-app --insecure --superlink {driver_addr} --dir {custom_dir} {self.server_app}"
        self._app_process = self._start_process("server-app", app_cmd)
        if not self._app_process:
            # stop the superlink
            self._superlink_process.kill()
            self._superlink_process = None
            raise RuntimeError("cannot start server_app process")

    @staticmethod
    def _stop_process(p):
        if p:
            try:
                p.kill()
            except:
                pass

    def stop(self, timeout=0.0):
        self._stop_process(self._app_process)
        self._app_process = None

        self._stop_process(self._superlink_process)
        self._superlink_process = None

    @staticmethod
    def _is_process_stopped(p):
        if p:
            return_code = p.poll()
            if return_code is None:
                return False, 0
            else:
                return True, return_code
        else:
            return True, 0

    def is_stopped(self) -> (bool, int):
        if self._start_error:
            return True, TieConstant.EXIT_CODE_CANT_START

        # check server app
        app_stopped, app_rc = self._is_process_stopped(self._app_process)
        if app_stopped:
            self._app_process = None

        superlink_stopped, superlink_rc = self._is_process_stopped(self._superlink_process)
        if superlink_stopped:
            self._superlink_process = None

        if app_stopped or superlink_stopped:
            self.stop()

        if app_stopped:
            return True, app_rc
        elif superlink_stopped:
            return True, superlink_rc
        else:
            return False, 0

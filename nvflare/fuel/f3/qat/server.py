# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.net_manager import NetManager
from nvflare.fuel.f3.mpm import MainProcessMonitor
from nvflare.fuel.hci.server.builtin import new_command_register_with_builtin_module
from nvflare.fuel.hci.server.hci import AdminServer
from nvflare.fuel.hci.server.login import LoginModule, SessionManager
from nvflare.fuel.utils.log_utils import get_obj_logger

from .cell_runner import CellRunner


class Server(CellRunner):
    def __init__(self, config_path: str, config_file: str, log_level: str):
        self._name = self.__class__.__name__
        self.logger = get_obj_logger(self)

        CellRunner.__init__(
            self, config_path=config_path, config_file=config_file, my_name=FQCN.ROOT_SERVER, log_level=log_level
        )

        net_mgr = NetManager(self.agent, diagnose=True)

        # set up admin server
        cmd_reg = new_command_register_with_builtin_module(app_ctx=self)
        sess_mgr = SessionManager(self.cell)
        login_module = LoginModule(sess_mgr)
        cmd_reg.register_module(login_module)
        cmd_reg.register_module(sess_mgr)
        cmd_reg.register_module(net_mgr)
        self.sess_mgr = sess_mgr

        self.admin = AdminServer(cmd_reg=cmd_reg, cell=self.cell, engine=None)

        MainProcessMonitor.add_cleanup_cb(self._clean_up)

    def start(self, start_all=True):
        super().start(start_all)
        self.admin.start()

    def _clean_up(self):
        # self.sess_mgr.shutdown()
        self.logger.debug(f"{self.cell.get_fqcn()}: Closed session manager")
        self.admin.stop()
        self.logger.debug(f"{self.cell.get_fqcn()}: Stopped Admin Server")

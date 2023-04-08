# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import threading
import time

from requests import Response

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.overseer_spec import SP, OverseerAgent


class DummyOverseerAgent(OverseerAgent):
    SSID = "ebc6125d-0a56-4688-9b08-355fe9e4d61a"

    def __init__(self, sp_end_point, heartbeat_interval=5):
        super().__init__()
        self._base_init(sp_end_point)

        self._report_and_query = threading.Thread(target=self._rnq_worker, args=())
        self._report_and_query.daemon = True
        self._flag = threading.Event()
        self._asked_to_exit = False
        self._update_callback = None
        self._conditional_cb = False
        self._heartbeat_interval = heartbeat_interval

    def _base_init(self, sp_end_point):
        self.sp_end_point = sp_end_point
        name, fl_port, admin_port = self.sp_end_point.split(":")
        self._psp = SP(name, fl_port, admin_port, DummyOverseerAgent.SSID, True)
        psp_dict = {
            "sp_end_point": sp_end_point,
            "service_session_id": DummyOverseerAgent.SSID,
            "primary": True,
            "state": "online",
        }
        self.overseer_info = {
            "primary_sp": psp_dict,
            "sp_list": [psp_dict],
            "system": "ready",
        }

    def initialize(self, fl_ctx: FLContext):
        sp_end_point = fl_ctx.get_prop(FLContextKey.SP_END_POINT)
        if sp_end_point:
            self._base_init(sp_end_point)

    def is_shutdown(self) -> bool:
        """Return whether the agent receives a shutdown request."""
        return False

    def get_primary_sp(self) -> SP:
        """Return current primary service provider. The PSP is static in the dummy agent."""
        return self._psp

    def promote_sp(self, sp_end_point, headers=None) -> Response:
        # a hack to create dummy response
        resp = Response()
        resp.status_code = 200
        resp._content = json.dumps({"Error": "this functionality is not supported by the dummy agent"}).encode("utf-8")
        return resp

    def start(self, update_callback=None, conditional_cb=False):
        self._conditional_cb = conditional_cb
        self._update_callback = update_callback
        self._report_and_query.start()
        self._flag.set()

    def pause(self):
        self._flag.clear()

    def resume(self):
        self._flag.set()

    def set_state(self, state) -> Response:
        # a hack to create dummy response
        resp = Response()
        resp.status_code = 200
        resp._content = json.dumps({"Error": "this functionality is not supported by the dummy agent"}).encode("utf-8")
        return resp

    def end(self):
        self._flag.set()
        self._asked_to_exit = True
        # self._report_and_query.join()

    def set_secure_context(self, ca_path: str, cert_path: str = "", prv_key_path: str = ""):
        pass

    def _do_callback(self):
        if self._update_callback:
            self._update_callback(self)

    def _rnq_worker(self):
        while not self._asked_to_exit:
            self._flag.wait()
            if not self._conditional_cb:
                self._do_callback()
            time.sleep(self._heartbeat_interval)

# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from requests.models import Response

from nvflare.apis.overseer_spec import SP, OverseerAgent


class DummyOverseerAgent(OverseerAgent):
    def __init__(self, sp_end_point, heartbeat_interval=5):
        name, fl_port, admin_port = sp_end_point.split(":")
        self._psp = SP(name, fl_port, admin_port, True)

        self._report_and_query = threading.Thread(target=self._rnq_worker, args=())
        self._flag = threading.Event()
        self._update_callback = None
        self._conditional_cb = False
        self._heartbeat_interval = heartbeat_interval

    def get_primary_sp(self) -> SP:
        """Return current primary service provider. The PSP is static in the dummy agent."""
        return self._psp

    def promote_sp(self, sp_end_point, headers=None):
        resp = Response()
        resp.status_code = 200
        resp.content = json.dumps({"error": "this functionality is not supported by the dummy agent"})
        return resp

    def start(self, update_callback=None, conditional_cb=False):
        self.conditional_cb = conditional_cb
        self._update_callback = update_callback
        self._report_and_query.start()
        self._flag.set()

    def pause(self):
        self._flag.clear()

    def resume(self):
        self._flag.set()

    def end(self):
        self._flag.set()
        self._asked_to_exit = True
        self._report_and_query.join()

    def _do_callback(self):
        if self._update_callback:
            self._update_callback(self)

    def _rnq_worker(self):
        while not self._asked_to_exit:
            self._flag.wait()
            if not self.conditional_cb:
                self._do_callback()
            time.sleep(self._heartbeat_interval)

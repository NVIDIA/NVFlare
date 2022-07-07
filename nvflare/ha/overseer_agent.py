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

import logging
import threading
import time
from typing import Any, Dict, Optional

from requests import Request, RequestException, Session, codes
from requests.adapters import HTTPAdapter

from nvflare.apis.overseer_spec import SP, OverseerAgent


class HttpOverseerAgent(OverseerAgent):
    def __init__(
        self,
        role,
        overseer_end_point,
        project,
        name: str,
        fl_port: str = "",
        admin_port: str = "",
        heartbeat_interval=5,
    ):
        if role not in ["server", "client", "admin"]:
            raise ValueError(f'Expect role in ["server", "client", "admin"] but got {role}')
        self._role = role
        self._overseer_end_point = overseer_end_point
        self._project = project
        self._session = None
        self._status_lock = threading.Lock()
        self._report_and_query = threading.Thread(target=self._rnq_worker, args=())
        self._psp = SP()
        self._flag = threading.Event()
        self._ca_path = None
        self._cert_path = None
        self._prv_key_path = None
        self._last_service_session_id = ""
        self._asked_to_exit = False
        self._logger = logging.getLogger(self.__class__.__name__)
        self._retry_delay = 4
        self._asked_to_stop_retrying = False
        self._overseer_info = {}
        self._update_callback = None
        self._conditional_cb = False
        if self._role == "server":
            self._sp_end_point = ":".join([name, fl_port, admin_port])
        self._heartbeat_interval = heartbeat_interval

    def _send(
        self, api_point, headers: Optional[Dict[str, Any]] = None, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try_count = 0
        while not self._asked_to_stop_retrying:
            try:
                req = Request("POST", api_point, json=payload, headers=headers)
                prepared = self._session.prepare_request(req)
                resp = self._session.send(prepared)
                return resp
            except RequestException as e:
                try_count += 1
                # self._logger.info(f"tried: {try_count} with exception: {e}")
                time.sleep(self._retry_delay)

    def set_secure_context(self, ca_path: str, cert_path: str = "", prv_key_path: str = ""):
        self._ca_path = ca_path
        self._cert_path = cert_path
        self._prv_key_path = prv_key_path

    def start(self, update_callback=None, conditional_cb=False):
        self._session = Session()
        adapter = HTTPAdapter(max_retries=1)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        if self._ca_path:
            self._session.verify = self._ca_path
            self._session.cert = (self._cert_path, self._prv_key_path)
        self.conditional_cb = conditional_cb
        if update_callback:
            self._update_callback = update_callback
        self._report_and_query.start()
        self._flag.set()

    def pause(self):
        self._asked_to_stop_retrying = True
        self._flag.clear()

    def resume(self):
        self._asked_to_stop_retrying = False
        self._flag.set()

    def end(self):
        self._asked_to_stop_retrying = True
        self._flag.set()
        self._asked_to_exit = True
        self._report_and_query.join()

    def is_shutdown(self) -> bool:
        """Return whether the agent receives a shutdown request."""
        return self._overseer_info.get("system") == "shutdown"

    def get_primary_sp(self) -> SP:
        """Return current primary service provider.

        If primary sp not available, such as not reported by SD, connection to SD not established yet
        the name and ports will be empty strings.
        """
        return self._psp

    def promote_sp(self, sp_end_point, headers=None):
        api_point = self._overseer_end_point + "/promote"
        return self._send(api_point, headers=None, payload={"sp_end_point": sp_end_point, "project": self._project})

    def set_state(self, state):
        api_point = self._overseer_end_point + "/state"
        return self._send(api_point, payload={"state": state})

    def _do_callback(self):
        if self._update_callback:
            self._update_callback(self)

    def _handle_ssid(self, ssid):
        if not self.conditional_cb or self._last_service_session_id != ssid:
            self._last_service_session_id = ssid
            self._do_callback()

    def _prepare_data(self):
        data = dict(role=self._role, project=self._project)
        return data

    def _rnq_worker(self):
        data = self._prepare_data()
        if self._role == "server":
            data["sp_end_point"] = self._sp_end_point
        api_point = self._overseer_end_point + "/heartbeat"
        while not self._asked_to_exit:
            self._flag.wait()
            self._rnq(api_point, headers=None, data=data)
            time.sleep(self._heartbeat_interval)

    def _rnq(self, api_point, headers, data):
        resp = self._send(api_point, headers=headers, payload=data)
        if resp is None:
            return
        if resp.status_code != codes.ok:
            return
        self._overseer_info = resp.json()
        psp = self._overseer_info.get("primary_sp")
        if psp:
            name, fl_port, admin_port = psp.get("sp_end_point").split(":")
            service_session_id = psp.get("service_session_id", "")
            self._psp = SP(name, fl_port, admin_port, service_session_id, True)
            # last_heartbeat = psp.get("last_heartbeat", "")
            self._handle_ssid(service_session_id)
        else:
            self._psp = SP()
            service_session_id = ""
            self._handle_ssid(service_session_id)

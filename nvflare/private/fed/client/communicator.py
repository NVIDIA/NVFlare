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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, ReturnCode
from nvflare.private.defs import CellChannel, SessionTopic, MessagePayloadKey
from nvflare.private.fed.rcmi import RootCellMessageInterface


class Communicator:
    def __init__(
        self,
        engine,
    ):
        """To init the Communicator.

        Args:
            engine:
        """
        self.engine = engine

        self.should_stop = False
        self.heartbeat_done = False
        self.logger = logging.getLogger(self.__class__.__name__)

    def client_registration(self, client_name, project_name):
        """Client's metadata used to authenticate and communicate.

        Args:
            client_name: client name
            project_name: FL study project name

        Returns:
            The client's token

        """
        cmi = self.engine.get_cmi()
        assert isinstance(cmi, RootCellMessageInterface)
        reply = cmi.send_to_server(
            channel=CellChannel.SESSION,
            topic=SessionTopic.REGISTER,
            timeout=1.0,
            headers={
                RootCellMessageInterface.HEADER_CLIENT_NAME: client_name,
                RootCellMessageInterface.HEADER_PROJECT_NAME: project_name
            },
            payload=None,
            fl_ctx=None
        )

        assert isinstance(reply, Shareable)
        rc = reply.get_return_code()
        if rc == ReturnCode.OK:
            token = reply.get(cmi.HEADER_CLIENT_TOKEN)
            ssid = reply.get(cmi.HEADER_SSID)
            self.should_stop = False
            return token, ssid
        else:
            self.logger.error(f"failed to register: {rc}")
            return None, None

    def quit_remote(self, project_name, token, fl_ctx: FLContext):
        """Sending the last message to the server before leaving.

        Args:
            project_name: project name
            token: FL client token
            fl_ctx: FLContext

        Returns:
            server's reply to the last message

        """
        cmi = self.engine.get_cmi()
        assert isinstance(cmi, RootCellMessageInterface)
        reply = cmi.send_to_server(
            channel=CellChannel.SESSION,
            topic=SessionTopic.LOGOUT,
            timeout=1.0,
            headers={
                RootCellMessageInterface.HEADER_CLIENT_TOKEN: token,
                RootCellMessageInterface.HEADER_PROJECT_NAME: project_name
            },
            payload=None,
            fl_ctx=fl_ctx
        )
        assert isinstance(reply, Shareable)
        rc = reply.get_return_code()
        if rc == ReturnCode.OK:
            self.logger.info("logged out successfully")
        else:
            self.logger.error(f"failed to logout: {rc}")

    def send_heartbeat(self, project_name, token, ssid, client_name):
        payload = Shareable(
            {
                MessagePayloadKey.JOBS: self.engine.get_all_job_ids()
            }
        )
        cmi = self.engine.get_cmi()
        assert isinstance(cmi, RootCellMessageInterface)
        reply = cmi.send_to_server(
            channel=CellChannel.SESSION,
            topic=SessionTopic.HEARTBEAT,
            timeout=1.0,
            headers={
                RootCellMessageInterface.HEADER_CLIENT_NAME: client_name,
                RootCellMessageInterface.HEADER_SSID: ssid,
                RootCellMessageInterface.HEADER_CLIENT_TOKEN: token,
                RootCellMessageInterface.HEADER_PROJECT_NAME: project_name
            },
            payload=payload,
            fl_ctx=None
        )
        assert isinstance(reply, Shareable)
        rc = reply.get_return_code()
        if rc == ReturnCode.OK:
            jobs_to_abort = reply.get(MessagePayloadKey.ABORT_JOBS)
            if jobs_to_abort:
                self._clean_up_runs(jobs_to_abort)
        else:
            self.logger.error(f"failed heartbeat: {rc}")

    def _clean_up_runs(self, jobs_to_abort):
        display_runs = ",".join(jobs_to_abort)
        try:
            if jobs_to_abort:
                for job in jobs_to_abort:
                    self.engine.abort_app(job)
                self.logger.info(f"These runs: {display_runs} are not running on the server. Aborted them.")
        except:
            self.logger.info(f"Failed to clean up the runs: {display_runs}")

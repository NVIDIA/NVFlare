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

from typing import Optional

import logging

from nvflare.apis.fl_constant import ServerCommandNames, ServerCommandKey, AdminCommandNames
from nvflare.apis.shareable import Shareable, simple_shareable, get_shareable_data
from nvflare.private.defs import CellChannel
from nvflare.private.fed.rcmi import RootCellMessageInterface

from nvflare.private.fed.server.run_manager import RunInfo


class RootCommander:

    def __init__(self, engine):
        self.engine = engine
        self.logger = logging.getLogger(self.__class__.__name__)

    def _send_command(self, fl_ctx, job_id: str, command_name: str, data: Shareable, fnf=False):
        cmi = self.engine.get_cmi()
        assert isinstance(cmi, RootCellMessageInterface)
        if fnf:
            timeout = 0
        else:
            timeout = 1.0

        return cmi.send_to_job_cell(
            job_id=job_id,
            channel=CellChannel.COMMAND,
            topic=command_name,
            headers=None,
            payload=data,
            fl_ctx=fl_ctx,
            timeout=timeout
        )

    def show_stats(self, job_id) -> dict:
        reply = None
        try:
            reply = self._send_command(
                job_id=job_id,
                command_name=ServerCommandNames.SHOW_STATS,
                data=simple_shareable(),
                fl_ctx=None
            )
        except BaseException:
            self.logger.error(f"Failed to show_stats for JOB: {job_id}")

        if reply is None:
            stats = {}
        else:
            stats = get_shareable_data(reply)
        return stats

    def get_errors(self, job_id) -> dict:
        reply = None
        try:
            reply = self._send_command(
                job_id=job_id,
                command_name=ServerCommandNames.GET_ERRORS,
                data=simple_shareable(),
                fl_ctx=None
            )
        except BaseException:
            self.logger.error(f"Failed to get_errors for JOB: {job_id}")

        if reply is None:
            errors = {}
        else:
            errors = get_shareable_data(reply)

        return errors

    def notify_dead_job(self, client_name: str, job_id: str):
        shareable = Shareable()
        shareable.set_header(ServerCommandKey.FL_CLIENT, client_name)
        self._send_command(
            job_id=job_id,
            command_name=ServerCommandNames.HANDLE_DEAD_JOB,
            data=shareable,
            fl_ctx=None,
            fnf=True
        )

    def abort(self, job_id: str):
        self._send_command(
            job_id=job_id,
            command_name=AdminCommandNames.ABORT,
            data=simple_shareable(),
            fl_ctx=None,
            fnf=True
        )

    def get_app_run_info(self, job_id) -> Optional[RunInfo]:
        reply = None
        try:
            reply = self._send_command(
                job_id=job_id,
                command_name=ServerCommandNames.GET_RUN_INFO,
                data=simple_shareable(),
                fl_ctx=None
            )
        except BaseException:
            self.logger.error(f"Failed to get_app_run_info for run: {job_id}")

        if reply:
            run_info = get_shareable_data(reply)
        else:
            run_info = None
        return run_info

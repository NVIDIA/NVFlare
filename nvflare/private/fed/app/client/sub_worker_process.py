# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""Sub_worker process to start the multi-processes client."""

import argparse
import copy
import logging
import os
import sys
import threading
import time

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.utils.fl_context_utils import get_serializable_data
from nvflare.apis.workspace import Workspace
from nvflare.app_common.executors.multi_process_executor import WorkerComponentBuilder
from nvflare.fuel.common.multi_process_executor_constants import (
    CommunicateData,
    CommunicationMetaData,
    MultiProcessCommandNames,
)
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.cell import Message as CellMessage
from nvflare.fuel.f3.cellnet.cell import MessageHeaderKey, make_reply
from nvflare.fuel.f3.cellnet.defs import ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.mpm import MainProcessMonitor as mpm
from nvflare.fuel.sec.audit import AuditService
from nvflare.fuel.sec.security_content_service import SecurityContentService
from nvflare.private.defs import CellChannel, CellChannelTopic, new_cell_message
from nvflare.private.fed.app.fl_conf import create_privacy_manager
from nvflare.private.fed.app.utils import monitor_parent_process
from nvflare.private.fed.client.client_run_manager import ClientRunManager
from nvflare.private.fed.runner import Runner
from nvflare.private.fed.simulator.simulator_app_runner import SimulatorClientRunManager
from nvflare.private.fed.utils.fed_utils import add_logfile_handler, configure_logging, fobs_initialize
from nvflare.private.privacy_manager import PrivacyService


class EventRelayer(FLComponent):
    """To relay the event from the worker_process."""

    def __init__(self, cell, parent_fqcn, local_rank):
        """To init the EventRelayer.

        Args:
            conn: worker_process connection.
            local_rank: process local rank
        """
        super().__init__()
        self.cell = cell
        self.parent_fqcn = parent_fqcn
        self.local_rank = local_rank

        self.event_lock = threading.Lock()
        self.start_run_fired = False

    def relay_event(self, run_manager, data):
        """To relay the event.

        Args:
            run_manager: Client_Run_Manager
            data: event data

        """
        with run_manager.new_context() as fl_ctx:
            event_type = data[CommunicationMetaData.EVENT_TYPE]
            if event_type == EventType.START_RUN:
                if self.start_run_fired:
                    return
                else:
                    self.start_run_fired = True
            fl_ctx.props.update(data[CommunicationMetaData.FL_CTX].props)

            fl_ctx.set_prop(
                FLContextKey.EVENT_ORIGIN_SITE, CommunicateData.MULTI_PROCESS_EXECUTOR, private=True, sticky=False
            )
            self.fire_event(event_type=event_type, fl_ctx=fl_ctx)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """To handle the event.

        Args:
            event_type: event_type
            fl_ctx: FLContext

        """
        event_site = fl_ctx.get_prop(FLContextKey.EVENT_ORIGIN_SITE)

        new_fl_ctx = FLContext()
        new_fl_ctx.props.update(copy.deepcopy(get_serializable_data(fl_ctx).props))
        if event_site != CommunicateData.MULTI_PROCESS_EXECUTOR:
            with self.event_lock:
                try:
                    data = {
                        CommunicationMetaData.EVENT_TYPE: event_type,
                        CommunicationMetaData.RANK_NUMBER: self.local_rank,
                        CommunicationMetaData.FL_CTX: new_fl_ctx,
                    }

                    request = new_cell_message({}, data)
                    return_data = self.cell.send_request(
                        target=self.parent_fqcn,
                        channel=CellChannel.MULTI_PROCESS_EXECUTOR,
                        topic=CellChannelTopic.FIRE_EVENT,
                        request=request,
                    )
                    # update the fl_ctx from the child process return data.
                    fl_ctx.props.update(return_data.payload[CommunicationMetaData.FL_CTX].props)
                except Exception:
                    self.log_warning(
                        fl_ctx, f"Failed to relay the event to parent process. Event: {event_type}", fire_event=False
                    )


class SubWorkerExecutor(Runner):
    def __init__(self, args, workspace, num_of_processes, local_rank) -> None:
        super().__init__()

        self.args = args
        self.workspace = workspace
        self.components = {}
        self.handlers = []
        self.executor = None
        self.run_manager = None
        self.num_of_processes = num_of_processes
        self.local_rank = local_rank

        self.done = False

        fqcn = FQCN.join([args.client_name, args.job_id, str(local_rank)])
        credentials = {}
        self.cell = Cell(
            fqcn=fqcn,
            root_url=args.root_url,
            secure=False,
            credentials=credentials,
            create_internal_listener=True,
            parent_url=args.parent_url,
        )
        self.cell.start()
        net_agent = NetAgent(self.cell)
        self.cell.register_request_cb(
            channel=CellChannel.CLIENT_SUB_WORKER_COMMAND,
            topic="*",
            cb=self.execute_command,
        )
        mpm.add_cleanup_cb(net_agent.close)
        mpm.add_cleanup_cb(self.cell.stop)

        self.commands = {
            MultiProcessCommandNames.INITIALIZE: self._initialize,
            MultiProcessCommandNames.TASK_EXECUTION: self._execute_task,
            MultiProcessCommandNames.FIRE_EVENT: self._handle_event,
            MultiProcessCommandNames.CLOSE: self._close,
        }

        self.logger = logging.getLogger(self.__class__.__name__)

    def execute_command(self, request: CellMessage) -> CellMessage:
        command_name = request.get_header(MessageHeaderKey.TOPIC)
        data = request.payload

        if command_name not in self.commands:
            return make_reply(ReturnCode.INVALID_REQUEST, "", None)
        return self.commands[command_name](data)

    def _initialize(self, data):
        executor_id = data[CommunicationMetaData.LOCAL_EXECUTOR]
        components_conf = data[CommunicationMetaData.COMPONENTS]
        component_builder = WorkerComponentBuilder()
        for item in components_conf:
            cid = item.get("id", None)
            if not cid:
                raise TypeError("missing component id")
            self.components[cid] = component_builder.build_component(item)
            if isinstance(self.components[cid], FLComponent):
                self.handlers.append(self.components[cid])

        self.executor = self.components.get(executor_id, None)
        if not isinstance(self.executor, Executor):
            make_reply(
                ReturnCode.INVALID_REQUEST,
                "invalid executor {}: expect Executor but got {}".format(executor_id, type(self.executor)),
                None,
            )

        job_id = self.args.job_id
        self._get_client_run_manager(job_id)

        parent_fqcn = FQCN.join([self.args.client_name, self.args.job_id])
        relayer = EventRelayer(self.cell, parent_fqcn, self.local_rank)
        self.run_manager.add_handler(relayer)
        self.run_manager.components[CommunicationMetaData.RELAYER] = relayer

        with self.run_manager.new_context() as fl_ctx:
            fl_ctx.set_prop(FLContextKey.RANK_NUMBER, self.local_rank, private=True, sticky=True)
            fl_ctx.set_prop(FLContextKey.NUM_OF_PROCESSES, self.num_of_processes, private=True, sticky=True)

            event_data = {
                CommunicationMetaData.EVENT_TYPE: EventType.START_RUN,
                CommunicationMetaData.FL_CTX: data[CommunicationMetaData.FL_CTX],
            }
            relayer.relay_event(self.run_manager, event_data)

        return make_reply(ReturnCode.OK, "", None)

    def _get_client_run_manager(self, job_id):
        if self.args.simulator_engine.lower() == "true":
            self.run_manager = SimulatorClientRunManager(
                client_name=self.args.client_name,
                job_id=job_id,
                workspace=self.workspace,
                client=None,
                components=self.components,
                handlers=self.handlers,
                conf=None,
            )
        else:
            self.run_manager = ClientRunManager(
                client_name=self.args.client_name,
                job_id=job_id,
                workspace=self.workspace,
                client=None,
                components=self.components,
                handlers=self.handlers,
                conf=None,
            )

    def _execute_task(self, data):
        """To execute the event task and pass to worker_process.

        Args:

        """
        with self.run_manager.new_context() as fl_ctx:
            abort_signal = Signal()

            task_name = data[CommunicationMetaData.TASK_NAME]
            shareable = data[CommunicationMetaData.SHAREABLE]
            fl_ctx.props.update(data[CommunicationMetaData.FL_CTX].props)

            shareable = self.executor.execute(
                task_name=task_name, shareable=shareable, fl_ctx=fl_ctx, abort_signal=abort_signal
            )

            if self.local_rank == 0:
                return_data = {
                    CommunicationMetaData.SHAREABLE: shareable,
                    CommunicationMetaData.FL_CTX: get_serializable_data(fl_ctx),
                }
                request = new_cell_message({}, return_data)
                fqcn = FQCN.join([self.args.client_name, self.args.job_id])
                self.cell.send_request(
                    target=fqcn,
                    channel=CellChannel.MULTI_PROCESS_EXECUTOR,
                    topic=CellChannelTopic.EXECUTE_RESULT,
                    request=request,
                )

    def _handle_event(self, data):
        """To handle the event.

        Args:

        """
        event_relayer = self.run_manager.get_component(CommunicationMetaData.RELAYER)
        event_relayer.relay_event(self.run_manager, data)

    def _close(self, data):
        self.done = True
        self.cell.stop()
        # mpm.stop()

    def run(self):
        self.logger.info("SubWorkerExecutor process started.")
        while not self.done:
            time.sleep(1.0)
        # self.cell.run()
        # mpm.run("Client sub_worker")
        self.logger.info("SubWorkerExecutor process shutdown.")

    def stop(self):
        self.done = True


def main():
    """Sub_worker process program."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-m", type=str, help="WORKSPACE folder", required=True)
    parser.add_argument("--num_processes", type=str, help="Listen ports", required=True)
    parser.add_argument("--job_id", "-n", type=str, help="job_id", required=True)
    parser.add_argument("--client_name", "-c", type=str, help="client name", required=True)
    parser.add_argument("--simulator_engine", "-s", type=str, help="simulator engine", required=True)
    parser.add_argument("--parent_pid", type=int, help="parent process pid", required=True)
    parser.add_argument("--root_url", type=str, help="root cell url", required=True)
    parser.add_argument("--parent_url", type=str, help="parent cell url", required=True)

    args = parser.parse_args()
    workspace = Workspace(args.workspace, args.client_name)
    app_custom_folder = workspace.get_client_custom_dir()
    if os.path.isdir(app_custom_folder):
        sys.path.append(app_custom_folder)
    configure_logging(workspace)

    fobs_initialize()

    SecurityContentService.initialize(content_folder=workspace.get_startup_kit_dir())

    # Initialize audit service since the job execution will need it!
    AuditService.initialize(workspace.get_audit_file_path())

    # configure privacy control!
    privacy_manager = create_privacy_manager(workspace, names_only=True)

    # initialize Privacy Service
    PrivacyService.initialize(privacy_manager)

    # local_rank = args.local_rank
    local_rank = int(os.environ["LOCAL_RANK"])
    num_of_processes = int(args.num_processes)
    sub_executor = SubWorkerExecutor(args, workspace, num_of_processes, local_rank)

    # start parent process checking thread
    parent_pid = args.parent_pid
    stop_event = threading.Event()
    thread = threading.Thread(target=monitor_parent_process, args=(sub_executor, parent_pid, stop_event))
    thread.start()

    job_id = args.job_id
    log_file = workspace.get_app_log_file_path(job_id)
    add_logfile_handler(log_file)

    sub_executor.run()
    AuditService.close()


if __name__ == "__main__":
    """
    This is the program for running rank processes in multi-process mode.
    """
    # main()
    mpm.run(main_func=main)

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

"""Sub_worker process to start the multi-processes client."""

import argparse
import copy
import os
import sys
import threading
import time
from multiprocessing.connection import Client, Listener

from nvflare.fuel.utils import fobs
from nvflare.apis.executor import Executor
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.f3.cellnet.defs import ReturnCode
from nvflare.fuel.f3.cellnet.cell import make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils.fl_context_utils import get_serializable_data
from nvflare.apis.workspace import Workspace
from nvflare.app_common.executors.multi_process_executor import WorkerComponentBuilder
from nvflare.fuel.common.multi_process_executor_constants import CommunicateData, CommunicationMetaData, MultiProcessCommandNames
from nvflare.fuel.sec.audit import AuditService
from nvflare.fuel.sec.security_content_service import SecurityContentService
from nvflare.private.fed.app.client.worker_process import check_parent_alive
from nvflare.private.fed.app.fl_conf import create_privacy_manager
from nvflare.private.fed.client.client_run_manager import ClientRunManager
from nvflare.private.fed.simulator.simulator_app_runner import SimulatorClientRunManager
from nvflare.private.fed.utils.fed_utils import add_logfile_handler, configure_logging, fobs_initialize
from nvflare.private.privacy_manager import PrivacyService
from nvflare.security.logging import secure_log_traceback
from nvflare.fuel.f3.cellnet.cell import Cell, Message as CellMessage, MessageHeaderKey
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.private.defs import CellChannel, new_cell_message


class EventRelayer(FLComponent):
    """To relay the event from the worker_process."""

    def __init__(self, conn, local_rank):
        """To init the EventRelayer.

        Args:
            conn: worker_process connection.
            local_rank: process local rank
        """
        super().__init__()
        self.conn = conn
        self.local_rank = local_rank

        self.event_lock = threading.Lock()

    def relay_event(self, run_manager, data):
        """To relay the event.

        Args:
            run_manager: Client_Run_Manager
            data: event data

        """
        with run_manager.new_context() as fl_ctx:
            event_type = data[CommunicationMetaData.EVENT_TYPE]
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
                    self.conn.send(data)

                    return_data = self.conn.recv()
                    # update the fl_ctx from the child process return data.
                    fl_ctx.props.update(return_data[CommunicationMetaData.FL_CTX].props)
                except BaseException:
                    self.log_warning(
                        fl_ctx, f"Failed to relay the event to parent process. Event: {event_type}", fire_event=False
                    )


class SubWorkerExecutor:

    def __init__(self, args, local_rank) -> None:
        self.components = {}
        self.handlers = []
        self.executor = None

        fqcn = FQCN.join([args.client_name, args.job_id, str(local_rank)])
        credentials = {}
        cell = Cell(
            fqcn=fqcn,
            root_url=args.root_url,
            secure=False,
            credentials=credentials,
            create_internal_listener=True,
            parent_url=args.parent_url,
        )
        cell.start()
        net_agent = NetAgent(cell)
        cell.register_request_cb(
            channel=CellChannel.CLIENT_SUB_WORKER_COMMAND,
            topic="*",
            cb=self.execute_command,
        )

        self.commands = {
            MultiProcessCommandNames.INITIALIZE: self.initialize
        }

    def execute_command(self, request: CellMessage) -> CellMessage:
        command_name = request.get_header(MessageHeaderKey.TOPIC)
        data = fobs.loads(request.payload)

        if command_name not in self.commands:
            return make_reply(ReturnCode.INVALID_REQUEST, "", None)
        return self.commands[command_name](data)

    def initialize(self, data):
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
            raise ValueError(
                "invalid executor {}: expect Executor but got {}".format(executor_id, type(self.executor))
            )

        return make_reply(ReturnCode.OK, "", None)


def main():
    """Sub_worker process program."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-m", type=str, help="WORKSPACE folder", required=True)
    # parser.add_argument("--parent_port", type=str, help="Parent listen port", required=True)
    parser.add_argument("--ports", type=str, help="Listen ports", required=True)
    # parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--job_id", "-n", type=str, help="job_id", required=True)
    parser.add_argument("--client_name", "-c", type=str, help="client name", required=True)
    parser.add_argument("--simulator_engine", "-s", type=str, help="simulator engine", required=True)
    parser.add_argument("--parent_pid", type=int, help="parent process pid", required=True)
    parser.add_argument("--root_url", type=str, help="root cell url", required=True)
    parser.add_argument("--parent_url", type=str, help="parent cell url", required=True)

    args = parser.parse_args()
    listen_ports = list(map(int, args.ports.split("-")))
    # parent_port = args.parent_port

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
    listen_port = listen_ports[local_rank * 3]
    exe_conn = _create_connection(listen_port)

    listen_port = listen_ports[local_rank * 3 + 1]
    handle_conn = _create_connection(listen_port)

    listen_port = listen_ports[local_rank * 3 + 2]

    # create_cell(args, local_rank)
    sub_executor = SubWorkerExecutor(args, local_rank)

    # start parent process checking thread
    parent_pid = args.parent_pid
    stop_event = threading.Event()
    thread = threading.Thread(target=check_parent_alive, args=(parent_pid, stop_event))
    thread.start()

    event_conn = None
    while not event_conn:
        try:
            address = ("localhost", listen_port)
            event_conn = Client(address, authkey=CommunicationMetaData.PARENT_PASSWORD.encode())
        except Exception:
            time.sleep(1.0)
            pass

    data = exe_conn.recv()

    # client_name = data[CommunicationMetaData.FL_CTX].get_prop(FLContextKey.CLIENT_NAME)
    # job_id = data[CommunicationMetaData.FL_CTX].get_prop(FLContextKey.CURRENT_RUN)
    # workspace = data[CommunicationMetaData.FL_CTX].get_prop(FLContextKey.WORKSPACE_OBJECT)
    job_id = args.job_id

    if args.simulator_engine.lower() == "true":
        run_manager = SimulatorClientRunManager(
            client_name=args.client_name,
            job_id=job_id,
            workspace=workspace,
            client=None,
            components=data[CommunicationMetaData.COMPONENTS],
            handlers=data[CommunicationMetaData.HANDLERS],
            conf=None,
        )
    else:
        run_manager = ClientRunManager(
            client_name=args.client_name,
            job_id=job_id,
            workspace=workspace,
            client=None,
            components=data[CommunicationMetaData.COMPONENTS],
            handlers=data[CommunicationMetaData.HANDLERS],
            conf=None,
        )

    log_file = workspace.get_app_log_file_path(job_id)
    add_logfile_handler(log_file)

    relayer = EventRelayer(event_conn, local_rank)
    run_manager.add_handler(relayer)
    run_manager.components[CommunicationMetaData.RELAYER] = relayer

    executor = data[CommunicationMetaData.LOCAL_EXECUTOR]
    exe_conn.send({CommunicationMetaData.RANK_PROCESS_STARTED: True})

    exe_thread = threading.Thread(target=execute, args=(run_manager, local_rank, exe_conn, executor))
    exe_thread.start()

    event_thread = threading.Thread(target=handle_event, args=(run_manager, local_rank, handle_conn))
    event_thread.start()

    with run_manager.new_context() as fl_ctx:
        fl_ctx.set_prop(FLContextKey.RANK_NUMBER, local_rank, private=True, sticky=True)
        num_of_processes = int(len(listen_ports) / 3)
        fl_ctx.set_prop(FLContextKey.NUM_OF_PROCESSES, num_of_processes, private=True, sticky=True)

    exe_thread.join()
    event_thread.join()
    AuditService.close()


# def create_cell(args, local_rank):
#     fqcn = FQCN.join([args.client_name, args.job_id, str(local_rank)])
#     credentials = {}
#     cell = Cell(
#         fqcn=fqcn,
#         root_url=args.root_url,
#         secure=False,
#         credentials=credentials,
#         create_internal_listener=True,
#         parent_url=args.parent_url,
#     )
#     cell.start()
#     net_agent = NetAgent(cell)
#     cell.register_request_cb(
#         channel=CellChannel.CLIENT_SUB_WORKER_COMMAND,
#         topic="*",
#         cb=execute_command,
#     )


def _create_connection(listen_port):
    address = ("localhost", int(listen_port))
    listener = Listener(address, authkey=CommunicationMetaData.CHILD_PASSWORD.encode())
    conn = listener.accept()
    return conn


def execute(run_manager, local_rank, exe_conn, executor):
    """To execute the event task and pass to worker_process.

    Args:
        run_manager: Client_Run_Manager
        local_rank: provcess local rank
        exe_conn: execution connection
        executor: local executor

    """
    try:
        abort_signal = None
        while True:
            data = exe_conn.recv()

            command = data[CommunicationMetaData.COMMAND]
            if command == CommunicateData.EXECUTE:
                with run_manager.new_context() as fl_ctx:
                    abort_signal = Signal()

                    task_name = data[CommunicationMetaData.TASK_NAME]
                    shareable = data[CommunicationMetaData.SHAREABLE]
                    fl_ctx.props.update(data[CommunicationMetaData.FL_CTX].props)

                    shareable = executor.execute(
                        task_name=task_name, shareable=shareable, fl_ctx=fl_ctx, abort_signal=abort_signal
                    )
                    if local_rank == 0:
                        return_data = {
                            CommunicationMetaData.SHAREABLE: shareable,
                            CommunicationMetaData.FL_CTX: get_serializable_data(fl_ctx),
                        }
                        exe_conn.send(return_data)

            elif command == CommunicateData.CLOSE:
                if abort_signal:
                    abort_signal.trigger(True)
                break
    except Exception:
        secure_log_traceback()
        print("If you abort client you can ignore this exception.")


def handle_event(run_manager, local_rank, exe_conn):
    """To handle the event.

    Args:
        run_manager: Client_run_manager
        local_rank: process local rank
        exe_conn: execute connection

    """
    try:
        while True:
            data = exe_conn.recv()

            command = data[CommunicationMetaData.COMMAND]
            if command == CommunicateData.HANDLE_EVENT:
                event_relayer = run_manager.get_component(CommunicationMetaData.RELAYER)
                event_relayer.relay_event(run_manager, data)

                fl_ctx = data[CommunicationMetaData.FL_CTX]
                if local_rank == 0:
                    return_data = {CommunicationMetaData.FL_CTX: get_serializable_data(fl_ctx)}
                    exe_conn.send(return_data)
            elif command == CommunicateData.CLOSE:
                break
    except Exception:
        secure_log_traceback()
        print("If you abort client you can ignore this exception.")


# def execute_command(request: CellMessage) -> CellMessage:
#     command_name = request.get_header(MessageHeaderKey.TOPIC)
#     data = fobs.loads(request.payload)
#     return new_cell_message({}, None)



if __name__ == "__main__":
    """
    This is the program for running rank processes in multi-process mode.
    """
    main()

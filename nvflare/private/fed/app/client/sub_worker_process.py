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
import logging
import os
import threading
import time
import traceback
from multiprocessing.connection import Client, Listener

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.utils.fl_context_utils import get_serializable_data
from nvflare.fuel.common.multi_process_executor_constants import CommunicateData, CommunicationMetaData
from nvflare.fuel.sec.security_content_service import SecurityContentService
from nvflare.private.fed.client.client_run_manager import ClientRunManager
from nvflare.private.fed.utils.fed_utils import add_logfile_handler


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


def main():
    """Sub_worker process program."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-m", type=str, help="WORKSPACE folder", required=True)
    # parser.add_argument("--parent_port", type=str, help="Parent listen port", required=True)
    parser.add_argument("--ports", type=str, help="Listen ports", required=True)
    # parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    listen_ports = list(map(int, args.ports.split("-")))
    # parent_port = args.parent_port

    startup = os.path.join(args.workspace, "startup")
    SecurityContentService.initialize(content_folder=startup)

    # local_rank = args.local_rank
    local_rank = int(os.environ["LOCAL_RANK"])
    listen_port = listen_ports[local_rank * 3]
    exe_conn = _create_connection(listen_port)

    listen_port = listen_ports[local_rank * 3 + 1]
    handle_conn = _create_connection(listen_port)

    listen_port = listen_ports[local_rank * 3 + 2]

    event_conn = None
    while not event_conn:
        try:
            address = ("localhost", listen_port)
            event_conn = Client(address, authkey=CommunicationMetaData.PARENT_PASSWORD.encode())
        except Exception as e:
            time.sleep(1.0)
            pass

    data = exe_conn.recv()

    client_name = data[CommunicationMetaData.FL_CTX].get_prop(FLContextKey.CLIENT_NAME)
    job_id = data[CommunicationMetaData.FL_CTX].get_prop(FLContextKey.CURRENT_RUN)
    workspace = data[CommunicationMetaData.FL_CTX].get_prop(FLContextKey.WORKSPACE_OBJECT)
    run_manager = ClientRunManager(
        client_name=client_name,
        job_id=job_id,
        workspace=workspace,
        client=None,
        components=data[CommunicationMetaData.COMPONENTS],
        handlers=data[CommunicationMetaData.HANDLERS],
        conf=None,
    )

    log_config_file_path = os.path.join(startup, "log.config")
    logging.config.fileConfig(fname=log_config_file_path, disable_existing_loggers=False)

    log_file = os.path.join(args.workspace, job_id, "log.txt")
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
        traceback.print_exc()
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
        traceback.print_exc()
        print("If you abort client you can ignore this exception.")


if __name__ == "__main__":
    """
    This is the program for running rank processes in multi-process mode.
    """
    main()

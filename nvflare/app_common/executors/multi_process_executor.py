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
import os
import shlex
import subprocess
import threading
import time
from abc import abstractmethod
from multiprocessing.connection import Client, Listener

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils.common_utils import get_open_ports
from nvflare.apis.utils.fl_context_utils import get_serializable_data
from nvflare.fuel.common.multi_process_executor_constants import CommunicateData, CommunicationMetaData
from nvflare.fuel.utils.class_utils import ModuleScanner
from nvflare.fuel.utils.component_builder import ComponentBuilder


class WorkerComponentBuilder(ComponentBuilder):
    FL_PACKAGES = ["nvflare"]
    FL_MODULES = ["client", "app"]

    def __init__(self) -> None:
        """Component to build workers."""
        super().__init__()
        self.module_scanner = ModuleScanner(WorkerComponentBuilder.FL_PACKAGES, WorkerComponentBuilder.FL_MODULES, True)

    def get_module_scanner(self):
        return self.module_scanner


class MultiProcessExecutor(Executor):
    def __init__(self, executor_id=None, num_of_processes=1, components=None):
        """Manage the multi-process execution life cycle.

        Arguments:
            executor_id: executor component ID
            num_of_processes: number of processes to create
            components: a dictionary for component classes to their arguments
        """
        super().__init__()
        self.executor_id = executor_id

        self.components = {}
        self.handlers = []
        self._build_components(components)

        if not isinstance(num_of_processes, int):
            raise TypeError("{} must be an instance of int but got {}".format(num_of_processes, type(num_of_processes)))
        if num_of_processes < 1:
            raise ValueError(f"{num_of_processes} must >= 1.")
        self.num_of_processes = num_of_processes
        self.executor = None

        self.logger = logging.getLogger(self.__class__.__name__)
        self.conn_clients = []
        self.exe_process = None
        self.open_ports = []

        self.stop_execute = False
        self.relay_threads = []
        self.finalized = False
        self.event_lock = threading.Lock()
        self.relay_lock = threading.Lock()

    @abstractmethod
    def get_multi_process_command(self) -> str:
        """Provide the command for starting multi-process execution.

        Returns:
            multi-process starting command
        """
        return ""

    def _build_components(self, components):
        component_builder = WorkerComponentBuilder()
        for item in components:
            cid = item.get("id", None)
            if not cid:
                raise TypeError("missing component id")
            self.components[cid] = component_builder.build_component(item)
            if isinstance(self.components[cid], FLComponent):
                self.handlers.append(self.components[cid])

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.END_RUN:
            self.finalize(fl_ctx)

        self._pass_event_to_rank_processes(event_type, fl_ctx)

    def _pass_event_to_rank_processes(self, event_type: str, fl_ctx: FLContext):
        event_site = fl_ctx.get_prop(FLContextKey.EVENT_ORIGIN_SITE)

        if event_site != CommunicateData.SUB_WORKER_PROCESS:
            with self.event_lock:
                try:
                    data = {
                        CommunicationMetaData.COMMAND: CommunicateData.HANDLE_EVENT,
                        CommunicationMetaData.FL_CTX: get_serializable_data(fl_ctx),
                        CommunicationMetaData.EVENT_TYPE: event_type,
                    }
                    # send the init data to all the child processes
                    for conn_client in self.conn_clients:
                        conn_client[CommunicationMetaData.HANDLE_CONN].send(data)

                    return_data = self.conn_clients[0][CommunicationMetaData.HANDLE_CONN].recv()
                    # update the fl_ctx from the child process return data.
                    fl_ctx.props.update(return_data[CommunicationMetaData.FL_CTX].props)
                except BaseException as e:
                    # Warning: Have to set fire_event=False, otherwise it will cause dead loop on the event handling!!!
                    self.log_warning(
                        fl_ctx, f"Failed to relay the event to child processes. Event: {event_type}", fire_event=False
                    )

    def initialize(self, fl_ctx: FLContext):
        self.executor = self.components.get(self.executor_id, None)
        if not isinstance(self.executor, Executor):
            raise ValueError(
                "invalid executor {}: expect Executor but got {}".format(self.executor_id, type(self.executor))
            )
        self._initialize_multi_process(fl_ctx)

    def _initialize_multi_process(self, fl_ctx: FLContext):

        try:
            self.open_ports = get_open_ports(self.num_of_processes * 3)

            command = (
                self.get_multi_process_command()
                + " -m nvflare.private.fed.app.client.sub_worker_process"
                + " -m "
                + fl_ctx.get_prop(FLContextKey.ARGS).workspace
                + " --ports "
                + "-".join([str(i) for i in self.open_ports])
            )
            self.logger.info(f"multi_process_executor command: {command}")
            # use os.setsid to create new process group ID
            self.exe_process = subprocess.Popen(shlex.split(command, " "), preexec_fn=os.setsid, env=os.environ.copy())

            for i in range(self.num_of_processes):
                listen_port = self.open_ports[i * 3 + 2]
                thread = threading.Thread(target=self._relay_fire_event, args=(listen_port, fl_ctx))
                self.relay_threads.append(thread)
                thread.start()

                open_port = self.open_ports[i * 3]
                exe_conn = self._create_connection(open_port)

                open_port = self.open_ports[i * 3 + 1]
                handle_conn = self._create_connection(open_port)

                self.conn_clients.append(
                    {CommunicationMetaData.EXE_CONN: exe_conn, CommunicationMetaData.HANDLE_CONN: handle_conn}
                )
            self.logger.info(f"Created the connections to child processes on ports: {str(self.open_ports)}")

            data = {
                CommunicationMetaData.FL_CTX: get_serializable_data(fl_ctx),
                CommunicationMetaData.COMPONENTS: self.components,
                CommunicationMetaData.HANDLERS: self.handlers,
                CommunicationMetaData.LOCAL_EXECUTOR: self.executor,
            }

            # send the init data to all the child processes
            responses = []
            for conn_client in self.conn_clients:
                conn_client[CommunicationMetaData.EXE_CONN].send(data)
                responses.append(False)

            while True:
                run_abort_signal = fl_ctx.get_run_abort_signal()
                if run_abort_signal and run_abort_signal.triggered:
                    self.finalize(fl_ctx)
                    break

                # Make sure to receive responses from all rank processes.
                index = 0
                received_all = True
                for conn_client in self.conn_clients:
                    received_all = received_all and responses[index]
                    if not responses[index]:
                        if conn_client[CommunicationMetaData.EXE_CONN].poll(0.2):
                            conn_client[CommunicationMetaData.EXE_CONN].recv()
                            responses[index] = True
                    index += 1
                if received_all:
                    break
        except:
            self.log_exception(fl_ctx, "error initializing multi_process executor")

    def _relay_fire_event(self, listen_port, fl_ctx: FLContext):
        address = ("localhost", int(listen_port))
        listener = Listener(address, authkey=CommunicationMetaData.PARENT_PASSWORD.encode())
        conn = listener.accept()

        while not self.stop_execute:
            try:
                if conn.poll(0.1):
                    data = conn.recv()
                    event_type = data[CommunicationMetaData.EVENT_TYPE]
                    rank_number = data[CommunicationMetaData.RANK_NUMBER]

                    with self.relay_lock:
                        fl_ctx.props.update(data[CommunicationMetaData.FL_CTX].props)

                        fl_ctx.set_prop(FLContextKey.FROM_RANK_NUMBER, rank_number, private=True, sticky=False)
                        fl_ctx.set_prop(
                            FLContextKey.EVENT_ORIGIN_SITE,
                            CommunicateData.SUB_WORKER_PROCESS,
                            private=True,
                            sticky=False,
                        )
                        engine = fl_ctx.get_engine()
                        engine.fire_event(event_type, fl_ctx)

                        return_data = {CommunicationMetaData.FL_CTX: get_serializable_data(fl_ctx)}
                    conn.send(return_data)
            except:
                self.logger.warning("Failed to relay the fired events from rank_processes.")

    def _create_connection(self, open_port):
        conn = None
        while not conn:
            try:
                address = ("localhost", open_port)
                conn = Client(address, authkey=CommunicationMetaData.CHILD_PASSWORD.encode())
            except BaseException as e:
                time.sleep(1.0)
                pass
        return conn

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if not self.executor:
            raise RuntimeError("There's no executor for task {}".format(task_name))

        return self._execute_multi_process(
            task_name=task_name, shareable=shareable, fl_ctx=fl_ctx, abort_signal=abort_signal
        )

    def _execute_multi_process(
        self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> Shareable:

        if abort_signal.triggered:
            self.finalize(fl_ctx)
            return make_reply(ReturnCode.OK)

        try:
            data = {
                CommunicationMetaData.COMMAND: CommunicateData.EXECUTE,
                CommunicationMetaData.TASK_NAME: task_name,
                CommunicationMetaData.SHAREABLE: shareable,
                CommunicationMetaData.FL_CTX: get_serializable_data(fl_ctx),
            }

            # send the execute command to all the child processes
            for conn_client in self.conn_clients:
                conn_client[CommunicationMetaData.EXE_CONN].send(data)

            while True:
                if abort_signal.triggered:
                    self.finalize(fl_ctx)
                    return make_reply(ReturnCode.OK)

                if self.conn_clients[0][CommunicationMetaData.EXE_CONN].poll(1.0):
                    # Only need to receive the Shareable and FLContext update from rank 0 process.
                    return_data = self.conn_clients[0][CommunicationMetaData.EXE_CONN].recv()
                    shareable = return_data[CommunicationMetaData.SHAREABLE]
                    fl_ctx.props.update(return_data[CommunicationMetaData.FL_CTX].props)
                    return shareable
        except BaseException as e:
            self.log_error(fl_ctx, "Multi-Process Execution error.")
            return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)

    def finalize(self, fl_ctx: FLContext):
        """This is called when exiting/aborting the executor."""
        if self.finalized:
            return

        self.finalized = True
        self.stop_execute = True

        data = {CommunicationMetaData.COMMAND: CommunicateData.CLOSE}
        for conn_client in self.conn_clients:
            try:
                conn_client[CommunicationMetaData.EXE_CONN].send(data)
                conn_client[CommunicationMetaData.HANDLE_CONN].send(data)
                self.logger.info("close command sent to rank processes.")
            except:
                self.logger.warning("Failed to send the close command. ")
        try:
            os.killpg(os.getpgid(self.exe_process.pid), 9)
            self.logger.debug("kill signal sent")
        except Exception as e:
            pass

        if self.exe_process:
            self.exe_process.terminate()

        # wait for all relay threads to join!
        for t in self.relay_threads:
            if t.is_alive():
                t.join()

        self.log_info(fl_ctx, "Multi-Process Executor finalized!", fire_event=False)

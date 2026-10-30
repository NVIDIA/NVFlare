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

import math
import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from abc import abstractmethod

import psutil

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ConfigVarName, FLContextKey, ReturnCode, SystemConfigs
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils.fl_context_utils import get_serializable_data
from nvflare.app_common.widgets.component_path_authorizer import ComponentPathAuthorizer
from nvflare.fuel.common.excepts import ComponentNotAuthorized
from nvflare.fuel.common.multi_process_executor_constants import (
    CommunicateData,
    CommunicationMetaData,
    MultiProcessCommandNames,
)
from nvflare.fuel.f3.cellnet.core_cell import Message as CellMessage
from nvflare.fuel.f3.cellnet.core_cell import MessageHeaderKey
from nvflare.fuel.f3.cellnet.core_cell import ReturnCode as F3ReturnCode
from nvflare.fuel.f3.cellnet.core_cell import make_reply as F3make_reply
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.utils.class_utils import ModuleScanner
from nvflare.fuel.utils.component_builder import ComponentBuilder
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.deprecated import warn_deprecated
from nvflare.fuel.utils.json_scanner import Node
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.private.defs import CellChannel, CellChannelTopic, new_cell_message
from nvflare.security.logging import secure_format_exception
from nvflare.utils.process_utils import popen_in_new_session


class WorkerComponentBuilder(ComponentBuilder):
    FL_PACKAGES = ["nvflare"]
    FL_MODULES = ["client", "app"]

    def __init__(self, fl_ctx: FLContext = None, workspace=None, enforce_authorization=True) -> None:
        """Component to build workers."""
        super().__init__()
        self.module_scanner = ModuleScanner(WorkerComponentBuilder.FL_PACKAGES, WorkerComponentBuilder.FL_MODULES, True)
        self.fl_ctx = fl_ctx
        self.workspace = workspace
        self.enforce_authorization = enforce_authorization
        self.component_path_authorizer = ComponentPathAuthorizer()

    def get_module_scanner(self):
        return self.module_scanner

    def has_component_build_authorizer(self):
        return self.enforce_authorization

    @staticmethod
    def make_component_node(component_config, index=None):
        """Create a config node for a worker component payload.

        Args:
            component_config: Component configuration dict.
            index: Optional 1-based index when the config came from a components list.
        """
        node = Node(component_config)
        node.key = "component"
        node.paths = ["component"]
        if index is not None:
            node.key = f"#{index}"
            node.paths = ["components", node.key]
        return node

    @staticmethod
    def _make_component_node(component_config, index=None):
        return WorkerComponentBuilder.make_component_node(component_config, index)

    @staticmethod
    def _make_child_node(parent_node, element, key):
        node = Node(element)
        node.processor = parent_node.processor
        node.parent = parent_node
        node.level = parent_node.level + 1
        node.key = str(key)
        node.paths = parent_node.paths.copy()
        node.paths.append(node.key)
        return node

    def _is_authorizable_component_config(self, config_dict, node=None):
        return self.is_authorizable_component_config(config_dict, node)

    def _authorize_component_config(self, config_dict, node):
        if not self.enforce_authorization:
            return
        try:
            self.component_path_authorizer.authorize_component_config(
                config_dict, node, fl_ctx=self.fl_ctx, workspace=self.workspace
            )
        except UnsafeComponentError as ex:
            raise ComponentNotAuthorized(f"component not authorized: {ex}")

    def _authorize_component_config_tree(self, element, node, force_current=False):
        if isinstance(element, dict):
            if force_current or self._is_authorizable_component_config(element, node):
                self._authorize_component_config(element, node)
                args = element.get("args")
                if isinstance(args, (dict, list)):
                    self._authorize_component_config_tree(args, self._make_child_node(node, args, "args"))
                return

            for key, value in element.items():
                if isinstance(value, (dict, list)):
                    self._authorize_component_config_tree(value, self._make_child_node(node, value, key))
        elif isinstance(element, list):
            for i, item in enumerate(element):
                if isinstance(item, (dict, list)):
                    self._authorize_component_config_tree(item, self._make_child_node(node, item, f"#{i + 1}"))

    def build_component(self, config_dict, node=None):
        if node is None:
            node = self.make_component_node(config_dict)
        self._authorize_component_config_tree(config_dict, node, force_current=True)
        return super().build_component(config_dict)


class MultiProcessExecutor(Executor):
    def __init__(self, executor_id=None, num_of_processes=1, components=None, task_timeout=None):
        """Manage the multi-process execution life cycle.

        Arguments:
            executor_id: executor component ID
            num_of_processes: number of processes to create
            components: a dictionary for component classes to their arguments
            task_timeout: optional positive task execution timeout in seconds
        """
        super().__init__()
        warn_deprecated(
            "MultiProcessExecutor and PTMultiProcessExecutor are deprecated as of NVFlare 2.10. "
            "Migrate distributed training to job-based Client API with external-process torchrun. "
            "See the Migration Guide for event and component compatibility.",
            stacklevel=3,
        )
        if task_timeout is not None and (
            isinstance(task_timeout, bool)
            or not isinstance(task_timeout, (int, float))
            or not math.isfinite(task_timeout)
            or task_timeout <= 0
        ):
            raise ValueError("task_timeout must be positive or None")
        self.task_timeout = task_timeout
        self.executor_id = executor_id

        self.components_conf = components
        self.components = {}
        self.handlers = []
        self._components_built = False

        if not isinstance(num_of_processes, int):
            raise TypeError("{} must be an instance of int but got {}".format(num_of_processes, type(num_of_processes)))
        if num_of_processes < 1:
            raise ValueError(f"{num_of_processes} must >= 1.")
        self.num_of_processes = num_of_processes
        self.executor = None
        self.execute_result = None
        self.execute_complete = None
        self.engine = None

        self.logger = get_obj_logger(self)
        self.conn_clients = []
        self.exe_process = None
        self.targets = []
        self._child_processes = {}
        self._launcher_process = None
        self._result_lock = threading.Lock()
        self._result_ready = threading.Event()
        self._execution_id = None
        self._pending_ranks = set()
        self._worker_failed = False
        self._graceful_shutdown = False
        self._finalize_lock = threading.Lock()

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

    def _build_components(self, components, fl_ctx: FLContext = None, workspace=None):
        component_builder = WorkerComponentBuilder(fl_ctx=fl_ctx, workspace=workspace)
        for i, item in enumerate(components):
            cid = item.get("id", None)
            if not cid:
                raise TypeError("missing component id")
            node = component_builder.make_component_node(item, i + 1)
            self.components[cid] = component_builder.build_component(item, node)
            if isinstance(self.components[cid], FLComponent):
                self.handlers.append(self.components[cid])

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.END_RUN:
            self._graceful_shutdown = not self.stop_execute and self._pass_event_to_rank_processes(
                event_type, fl_ctx, wait=True
            )
            self.finalize(fl_ctx)
            return

        self._pass_event_to_rank_processes(event_type, fl_ctx)

    def _pass_event_to_rank_processes(self, event_type: str, fl_ctx: FLContext, wait=False):
        event_site = fl_ctx.get_prop(FLContextKey.EVENT_ORIGIN_SITE)

        if self.engine:
            if event_site != CommunicateData.SUB_WORKER_PROCESS:
                with self.event_lock:
                    try:
                        data = {
                            CommunicationMetaData.COMMAND: CommunicateData.HANDLE_EVENT,
                            CommunicationMetaData.FL_CTX: get_serializable_data(fl_ctx),
                            CommunicationMetaData.EVENT_TYPE: event_type,
                        }
                        # send the init data to all the child processes
                        request = new_cell_message({}, data)
                        if wait:
                            replies = self.engine.client.cell.broadcast_request(
                                targets=self.targets,
                                channel=CellChannel.CLIENT_SUB_WORKER_COMMAND,
                                topic=MultiProcessCommandNames.FIRE_EVENT,
                                request=request,
                                timeout=5.0,
                            )
                            return all(
                                target in replies
                                and replies[target].get_header(MessageHeaderKey.RETURN_CODE) == F3ReturnCode.OK
                                for target in self.targets
                            )
                        self.engine.client.cell.fire_and_forget(
                            targets=self.targets,
                            channel=CellChannel.CLIENT_SUB_WORKER_COMMAND,
                            topic=MultiProcessCommandNames.FIRE_EVENT,
                            message=request,
                        )
                    except Exception:
                        # Warning: Have to set fire_event=False, otherwise it will cause dead loop on the event handling!!!
                        self.log_warning(
                            fl_ctx,
                            f"Failed to relay the event to child processes. Event: {event_type}",
                            fire_event=False,
                        )
        return False

    def initialize(self, fl_ctx: FLContext):
        if not self._components_built:
            self._build_components(
                self.components_conf,
                fl_ctx=fl_ctx,
                workspace=fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT),
            )
            self._components_built = True

        self.executor = self.components.get(self.executor_id, None)
        if not isinstance(self.executor, Executor):
            raise ValueError(
                "invalid executor {}: expect Executor but got {}".format(self.executor_id, type(self.executor))
            )
        self._initialize_multi_process(fl_ctx)

    def _initialize_multi_process(self, fl_ctx: FLContext):

        try:
            client_name = fl_ctx.get_identity_name()
            job_id = fl_ctx.get_job_id()

            decomposer_module = ConfigService.get_str_var(
                name=ConfigVarName.DECOMPOSER_MODULE, conf=SystemConfigs.RESOURCES_CONF
            )

            self.engine = fl_ctx.get_engine()
            simulate_mode = fl_ctx.get_prop(FLContextKey.SIMULATE_MODE, False)
            cell = self.engine.client.cell
            # Create the internal listener for grand child process
            cell.make_internal_listener()
            command = (
                self.get_multi_process_command()
                + " -m nvflare.private.fed.app.client.sub_worker_process"
                + " -m "
                + fl_ctx.get_prop(FLContextKey.ARGS).workspace
                + " -c "
                + client_name
                + " -n "
                + job_id
                + " --num_processes "
                + str(self.num_of_processes)
                + " --simulator_engine "
                + str(simulate_mode)
                + " --parent_pid "
                + str(os.getpid())
                + " --root_url "
                + str(cell.get_root_url_for_child())
                + " --parent_url "
                + str(cell.get_internal_listener_url())
                + " --decomposer_module "
                + str(decomposer_module)
            )
            self.logger.info(f"multi_process_executor command: {command}")
            # start in a new session to create new process group ID
            self.exe_process = popen_in_new_session(shlex.split(command, " "), shell=False, env=os.environ.copy())
            try:
                self._launcher_process = psutil.Process(self.exe_process.pid)
                self._launcher_process.create_time()
            except psutil.NoSuchProcess:
                self._launcher_process = None

            # send the init data to all the child processes
            cell.register_request_cb(
                channel=CellChannel.MULTI_PROCESS_EXECUTOR,
                topic=CellChannelTopic.EXECUTE_RESULT,
                cb=self.receive_execute_result,
            )
            cell.register_request_cb(
                channel=CellChannel.MULTI_PROCESS_EXECUTOR,
                topic=CellChannelTopic.FIRE_EVENT,
                cb=self._relay_fire_event,
            )

            self.targets = []
            for i in range(self.num_of_processes):
                fqcn = FQCN.join([cell.get_fqcn(), str(i)])
                start = time.monotonic()
                self._remember_child_processes()
                while not cell.is_cell_reachable(fqcn):
                    self._remember_child_processes()
                    if self.exe_process.poll() is not None:
                        raise RuntimeError("multi-process launcher exited during initialization")
                    run_abort = fl_ctx.get_run_abort_signal()
                    if self.stop_execute or (run_abort and run_abort.triggered):
                        raise RuntimeError("multi-process initialization aborted")
                    time.sleep(1.0)
                    if time.monotonic() - start > 60.0:
                        raise RuntimeError(f"Could not reach the communication cell: {fqcn}")
                self.targets.append(fqcn)
            self._remember_child_processes()
            request = new_cell_message(
                {},
                {
                    CommunicationMetaData.FL_CTX: get_serializable_data(fl_ctx),
                    CommunicationMetaData.COMPONENTS: self.components_conf,
                    CommunicationMetaData.LOCAL_EXECUTOR: self.executor_id,
                },
            )
            replies = cell.broadcast_request(
                targets=self.targets,
                channel=CellChannel.CLIENT_SUB_WORKER_COMMAND,
                topic=MultiProcessCommandNames.INITIALIZE,
                request=request,
                timeout=60.0,
            )
            for name in self.targets:
                reply = replies.get(name)
                if reply is None or reply.get_header(MessageHeaderKey.RETURN_CODE) != F3ReturnCode.OK:
                    raise RuntimeError(f"error initializing multi-process rank {name}")
        except Exception as e:
            self.log_exception(fl_ctx, f"error initializing multi_process executor: {secure_format_exception(e)}")
            self.finalize(fl_ctx)
            self.system_panic("multi-process executor initialization failed", fl_ctx)
            raise

    def receive_execute_result(self, request: CellMessage) -> CellMessage:
        return_data = request.payload
        with self._result_lock:
            if self.execute_complete or self.stop_execute:
                return F3make_reply(F3ReturnCode.INVALID_REQUEST)
            if return_data.get(CommunicationMetaData.EXECUTION_ID) != self._execution_id:
                return F3make_reply(F3ReturnCode.INVALID_REQUEST)
            rank = return_data.get(CommunicationMetaData.RANK_NUMBER)
            if rank not in self._pending_ranks:
                return F3make_reply(F3ReturnCode.INVALID_REQUEST)
            result = return_data[CommunicationMetaData.SHAREABLE]
            self._pending_ranks.remove(rank)
            if isinstance(result, Shareable) and result.get_return_code() != ReturnCode.OK:
                self.execute_result = result
                self._worker_failed = True
            elif rank == 0:
                self.execute_result = (
                    result if isinstance(result, Shareable) else make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
                )
                self._worker_failed = not isinstance(result, Shareable)
            if self._worker_failed or not self._pending_ranks:
                self.execute_complete = True
                self._result_ready.set()
        return F3make_reply(F3ReturnCode.OK)

    def _relay_fire_event(self, request: CellMessage) -> CellMessage:
        data = request.payload
        with self.engine.new_context() as fl_ctx:
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
                self.engine.fire_event(event_type, fl_ctx)
                return_data = {CommunicationMetaData.FL_CTX: get_serializable_data(fl_ctx)}
                return F3make_reply(ReturnCode.OK, "", return_data)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if not self.executor:
            raise RuntimeError("There's no executor for task {}".format(task_name))

        with self._result_lock:
            self.execute_complete = False
            self.execute_result = None
            self._worker_failed = False
            self._execution_id = uuid.uuid4().hex
            self._pending_ranks = set(range(self.num_of_processes))
            self._result_ready.clear()
        start = time.monotonic()
        result = self._execute_multi_process(task_name, shareable, fl_ctx, abort_signal)
        if result is not None:
            self.finalize(fl_ctx)
            return result
        while True:
            self._remember_child_processes()
            if abort_signal.triggered or self.stop_execute:
                result = make_reply(ReturnCode.TASK_ABORTED)
                break
            if self._result_ready.is_set():
                with self._result_lock:
                    result = self.execute_result
                    failed = self._worker_failed
                if not failed:
                    return result
                break
            if self.exe_process is not None and self.exe_process.poll() is not None:
                result = make_reply(ReturnCode.EXECUTION_EXCEPTION)
                break
            if self.task_timeout is not None and time.monotonic() - start >= self.task_timeout:
                result = make_reply(ReturnCode.TIMEOUT)
                break
            self._result_ready.wait(0.2)
        self.finalize(fl_ctx)
        return result

    def _execute_multi_process(
        self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> Shareable:

        if abort_signal.triggered or self.stop_execute:
            return make_reply(ReturnCode.TASK_ABORTED)

        self.engine = fl_ctx.get_engine()
        try:
            data = {
                CommunicationMetaData.COMMAND: CommunicateData.EXECUTE,
                CommunicationMetaData.TASK_NAME: task_name,
                CommunicationMetaData.SHAREABLE: shareable,
                CommunicationMetaData.FL_CTX: get_serializable_data(fl_ctx),
                CommunicationMetaData.EXECUTION_ID: self._execution_id,
            }

            request = new_cell_message({}, data)
            errors = self.engine.client.cell.fire_and_forget(
                targets=self.targets,
                channel=CellChannel.CLIENT_SUB_WORKER_COMMAND,
                topic=MultiProcessCommandNames.TASK_EXECUTION,
                message=request,
            )
            if any(errors.values()):
                return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
        except Exception:
            self.log_error(fl_ctx, "Multi-Process Execution error.")
            return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)

    def _remember_child_processes(self):
        # torchrun ranks may own separate sessions. Retain identity-checked
        # Process handles while their ancestry is still observable.
        if self._launcher_process is None:
            return
        try:
            if not self._launcher_process.is_running():
                return
            for child in self._launcher_process.children(recursive=True):
                child.create_time()
                self._child_processes[child.pid] = child
        except psutil.NoSuchProcess:
            pass

    def finalize(self, fl_ctx: FLContext):
        """This is called when exiting/aborting the executor."""
        with self._finalize_lock:
            if self.finalized:
                return
            self.finalized = True
            self.stop_execute = True
            self._result_ready.set()
            self._remember_child_processes()
            try:
                if self.engine and self.targets:
                    errors = self.engine.client.cell.fire_and_forget(
                        targets=self.targets,
                        channel=CellChannel.CLIENT_SUB_WORKER_COMMAND,
                        topic=MultiProcessCommandNames.CLOSE,
                        message=new_cell_message({}, None),
                    )
                    if any(errors.values()):
                        self._graceful_shutdown = False
            except Exception:
                self._graceful_shutdown = False
            process = self.exe_process
            if process is not None:
                run_abort = fl_ctx.get_run_abort_signal()
                if self._graceful_shutdown and not (run_abort and run_abort.triggered):
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        pass
                # Terminate ranks in their own sessions as well as the launcher.
                # psutil guards these operations against PID reuse.
                children = list(self._child_processes.values())
                # Kill each still-owned group before signalling individuals.
                # This also catches children forked after the ancestry snapshot,
                # and prevents a TERM handler from spawning an orphan on exit.
                for owner in [self._launcher_process, *children]:
                    if owner is None or not isinstance(owner.pid, int):
                        continue
                    try:
                        group = os.getpgid(owner.pid)
                        if group == owner.pid and owner.is_running():
                            os.killpg(group, signal.SIGKILL)
                    except (ProcessLookupError, psutil.NoSuchProcess):
                        pass
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                if children:
                    _, alive = psutil.wait_procs(children, timeout=2.0)
                    for child in alive:
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    _, alive = psutil.wait_procs(alive, timeout=2.0)
                    if alive:
                        self.log_warning(fl_ctx, "rank processes did not settle after termination", fire_event=False)
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
            for thread in self.relay_threads:
                thread.join(timeout=5.0)
            self.log_info(fl_ctx, "Multi-Process Executor finalized!", fire_event=False)

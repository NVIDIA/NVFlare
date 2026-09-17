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

import os
import shlex
import subprocess
import threading
import time
from abc import abstractmethod

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
from nvflare.fuel.utils.json_scanner import Node
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.private.defs import CellChannel, CellChannelTopic, new_cell_message
from nvflare.security.logging import secure_format_exception


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
    def __init__(self, executor_id=None, num_of_processes=1, components=None):
        """Manage the multi-process execution life cycle.

        Arguments:
            executor_id: executor component ID
            num_of_processes: number of processes to create
            components: a dictionary for component classes to their arguments
        """
        super().__init__()
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
            self.finalize(fl_ctx)

        self._pass_event_to_rank_processes(event_type, fl_ctx)

    def _pass_event_to_rank_processes(self, event_type: str, fl_ctx: FLContext):
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
            # use os.setsid to create new process group ID
            self.exe_process = subprocess.Popen(
                shlex.split(command, " "), shell=False, preexec_fn=os.setsid, env=os.environ.copy()
            )

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
                start = time.time()
                while not cell.is_cell_reachable(fqcn):
                    time.sleep(1.0)
                    if time.time() - start > 60.0:
                        raise RuntimeError(f"Could not reach the communication cell: {fqcn}")
                self.targets.append(fqcn)
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
            )
            for name, reply in replies.items():
                if reply.get_header(MessageHeaderKey.RETURN_CODE) != F3ReturnCode.OK:
                    self.log_exception(fl_ctx, "error initializing multi_process executor")
                    raise ValueError(reply.get_header(MessageHeaderKey.ERROR))
        except Exception as e:
            self.log_exception(fl_ctx, f"error initializing multi_process executor: {secure_format_exception(e)}")

    def receive_execute_result(self, request: CellMessage) -> CellMessage:
        return_data = request.payload
        with self.engine.new_context() as fl_ctx:
            fl_ctx.props.update(return_data[CommunicationMetaData.FL_CTX].props)
            self.execute_result = return_data[CommunicationMetaData.SHAREABLE]

        self.execute_complete = True
        return F3make_reply(ReturnCode.OK, "", None)

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

        self.execute_complete = False

        self._execute_multi_process(task_name=task_name, shareable=shareable, fl_ctx=fl_ctx, abort_signal=abort_signal)

        while not self.execute_complete:
            time.sleep(0.2)
        return self.execute_result

    def _execute_multi_process(
        self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> Shareable:

        if abort_signal.triggered:
            self.finalize(fl_ctx)
            return make_reply(ReturnCode.OK)

        self.engine = fl_ctx.get_engine()
        try:
            data = {
                CommunicationMetaData.COMMAND: CommunicateData.EXECUTE,
                CommunicationMetaData.TASK_NAME: task_name,
                CommunicationMetaData.SHAREABLE: shareable,
                CommunicationMetaData.FL_CTX: get_serializable_data(fl_ctx),
            }

            request = new_cell_message({}, data)
            self.engine.client.cell.fire_and_forget(
                targets=self.targets,
                channel=CellChannel.CLIENT_SUB_WORKER_COMMAND,
                topic=MultiProcessCommandNames.TASK_EXECUTION,
                message=request,
            )
        except Exception:
            self.log_error(fl_ctx, "Multi-Process Execution error.")
            return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)

    def finalize(self, fl_ctx: FLContext):
        """This is called when exiting/aborting the executor."""
        if self.finalized:
            return

        self.finalized = True
        self.stop_execute = True

        request = new_cell_message({}, None)
        self.engine.client.cell.fire_and_forget(
            targets=self.targets,
            channel=CellChannel.CLIENT_SUB_WORKER_COMMAND,
            topic=MultiProcessCommandNames.CLOSE,
            message=request,
        )

        try:
            os.killpg(os.getpgid(self.exe_process.pid), 9)
            self.logger.debug("kill signal sent")
        except Exception:
            pass

        if self.exe_process:
            self.exe_process.terminate()

        # wait for all relay threads to join!
        for t in self.relay_threads:
            if t.is_alive():
                t.join()

        self.log_info(fl_ctx, "Multi-Process Executor finalized!", fire_event=False)

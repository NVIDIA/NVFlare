# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from nvflare.apis.client import Client, from_dict
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.fox.api.app import ClientApp
from nvflare.fox.api.constants import MAKE_CLIENT_APP_METHOD, BackendType
from nvflare.fox.api.proxy_utils import create_proxy_with_children
from nvflare.fox.utils.decomposers import register_available_decomposers
from nvflare.fuel.f3.cellnet.fqcn import FQCN

from .adaptor import FoxAdaptor
from .backend import FlareBackend
from .constants import SYNC_TASK_NAME, SyncKey
from .subprocess_launcher import SubprocessLauncher
from .utils import prepare_for_remote_call, prepare_for_subprocess_call
from .ws import FlareWorkspace


class FoxExecutor(Executor, FoxAdaptor):
    """Fox Executor for client-side execution.

    Supports two execution modes:
    - inprocess=True (default): Execute @fox.collab methods in the same process
    - inprocess=False: Spawn a subprocess for execution (e.g., for torchrun DDP)

    When inprocess=False, the executor spawns a FoxWorker subprocess using the
    specified launcher command (e.g., "torchrun --nproc_per_node=4"). The worker
    connects back to the executor via CellNet and handles the actual training.
    """

    def __init__(
        self,
        client_obj_id: str,
        collab_obj_ids: List[str] = None,
        incoming_call_filters=None,
        outgoing_call_filters=None,
        incoming_result_filters=None,
        outgoing_result_filters=None,
        props: Dict[str, Any] = None,
        resource_dirs: Dict[str, str] = None,
        max_call_threads=100,
        # Subprocess execution options
        inprocess: bool = True,
        run_cmd: Optional[str] = None,
        training_module: Optional[str] = None,
        subprocess_timeout: float = 300.0,
    ):
        """Initialize FoxExecutor.

        Args:
            client_obj_id: ID of the client object component.
            collab_obj_ids: IDs of additional collab objects.
            incoming_call_filters: Filters for incoming calls.
            outgoing_call_filters: Filters for outgoing calls.
            incoming_result_filters: Filters for incoming results.
            outgoing_result_filters: Filters for outgoing results.
            props: Additional properties.
            resource_dirs: Resource directories.
            max_call_threads: Maximum threads for call handling.
            inprocess: If True, execute in-process. If False, use subprocess.
            run_cmd: Command to run the training subprocess
                (e.g., "torchrun --nproc_per_node=4").
            training_module: Python module containing @fox.collab methods
                (required when inprocess=False).
            subprocess_timeout: Timeout for subprocess operations.
        """
        Executor.__init__(self)
        FoxAdaptor.__init__(
            self,
            collab_obj_ids=collab_obj_ids,
            props=props,
            resource_dirs=resource_dirs,
            incoming_call_filters=incoming_call_filters,
            outgoing_call_filters=outgoing_call_filters,
            incoming_result_filters=incoming_result_filters,
            outgoing_result_filters=outgoing_result_filters,
        )
        self.client_obj_id = client_obj_id
        self.register_event_handler(EventType.START_RUN, self._handle_start_run)
        self.register_event_handler(EventType.END_RUN, self._handle_end_run)
        self.client_app = None
        self.client_ctx = None
        self.thread_executor = ThreadPoolExecutor(max_workers=max_call_threads, thread_name_prefix="fox_call")

        # Subprocess execution options
        self.inprocess = inprocess
        self.run_cmd = run_cmd
        self.training_module = training_module
        self.subprocess_timeout = subprocess_timeout
        self._subprocess_launcher: Optional[SubprocessLauncher] = None

        # Validate subprocess options
        if not inprocess and not training_module:
            raise ValueError("training_module is required when inprocess=False")

    def _handle_start_run(self, event_type: str, fl_ctx: FLContext):
        # Register PyTorch TensorDecomposer for tensor serialization over CellNet
        register_available_decomposers()

        fl_ctx.set_prop(FLContextKey.FOX_MODE, True, private=True, sticky=True)
        engine = fl_ctx.get_engine()
        client_obj = engine.get_component(self.client_obj_id)
        if not client_obj:
            self.system_panic(f"cannot get client component {self.client_obj_id}", fl_ctx)
            return

        client_name = fl_ctx.get_identity_name()

        app = ClientApp(client_obj)

        # If the app contains "make_client_app" method, call it to make the app instance!
        # Check both the ClientApp and its underlying object for the method
        make_client_app_f = getattr(app, MAKE_CLIENT_APP_METHOD, None)
        if not make_client_app_f:
            make_client_app_f = getattr(app.obj, MAKE_CLIENT_APP_METHOD, None)
        if make_client_app_f and callable(make_client_app_f):
            app = make_client_app_f(client_name, BackendType.FLARE)
            if not isinstance(app, ClientApp):
                raise RuntimeError(f"result returned by {MAKE_CLIENT_APP_METHOD} must be ClientApp but got {type(app)}")

        app.name = client_name
        app.set_backend_type(BackendType.FLARE)
        self.client_app = app

        err = self.process_config(self.client_app, fl_ctx)
        if err:
            self.system_panic(err, fl_ctx)

    def _handle_end_run(self, event_type: str, fl_ctx: FLContext):
        # Stop subprocess if running
        if self._subprocess_launcher:
            self.logger.info("Stopping subprocess worker...")
            self._subprocess_launcher.stop()
            self._subprocess_launcher = None

        if self.client_ctx:
            self.logger.debug(f"finalizing client app {self.client_app.name}")
            self.client_app.finalize(self.client_ctx)
        self.thread_executor.shutdown(wait=True, cancel_futures=True)

    def _prepare_server_proxy(self, job_id, cell, collab_interface: dict, abort_signal, fl_ctx: FLContext):
        server_name = "server"
        target_fqcn = FQCN.join([FQCN.ROOT_SERVER, job_id])
        backend = FlareBackend(
            manager=self,
            engine=fl_ctx.get_engine(),
            caller=self.client_app.name,
            cell=cell,
            target_fqcn=target_fqcn,
            abort_signal=abort_signal,
            thread_executor=self.thread_executor,
        )

        # Build child specs (same backend for all, different interfaces)
        child_specs = {}
        for name, itf in collab_interface.items():
            if name == "":
                continue
            child_specs[name] = {"interface": itf}  # Uses main_backend by default

        return create_proxy_with_children(
            app=self.client_app,
            target_name=server_name,
            target_fqn=target_fqcn,
            main_backend=backend,
            main_interface=collab_interface.get(""),
            child_specs=child_specs,
        )

    def _prepare_client_proxy(self, job_id, cell, client: Client, abort_signal, collab_interface, fl_ctx: FLContext):
        target_fqcn = FQCN.join([client.get_fqcn(), job_id])
        backend = FlareBackend(
            manager=self,
            engine=fl_ctx.get_engine(),
            caller=self.client_app.name,
            cell=cell,
            target_fqcn=target_fqcn,
            abort_signal=abort_signal,
            thread_executor=self.thread_executor,
        )

        # Build child specs (same backend for all, different interfaces)
        child_specs = {}
        collab_objs = self.client_app.get_collab_objects()
        if collab_objs:
            for name in collab_objs.keys():
                child_specs[name] = {"interface": collab_interface.get(name)}

        return create_proxy_with_children(
            app=self.client_app,
            target_name=client.name,
            target_fqn=target_fqcn,
            main_backend=backend,
            main_interface=collab_interface.get(""),
            child_specs=child_specs,
        )

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name != SYNC_TASK_NAME:
            self.log_error(fl_ctx, f"received unsupported task {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)

        server_collab_interface = shareable.get(SyncKey.COLLAB_INTERFACE)
        client_collab_interface = self.client_app.get_collab_interface()
        self.log_debug(fl_ctx, f"{client_collab_interface=} {server_collab_interface=}")

        engine = fl_ctx.get_engine()
        cell = engine.get_cell()
        client_name = fl_ctx.get_identity_name()

        if self.inprocess:
            # In-process mode: execute methods directly
            prepare_for_remote_call(
                cell=cell,
                app=self.client_app,
                logger=self.logger,
            )
        else:
            # Subprocess mode: spawn worker and forward calls
            self._start_subprocess_worker(cell, client_name)
            prepare_for_subprocess_call(
                cell=cell,
                app=self.client_app,
                subprocess_launcher=self._subprocess_launcher,
                logger=self.logger,
            )

        # build proxies
        job_id = fl_ctx.get_job_id()
        server_proxy = self._prepare_server_proxy(job_id, cell, server_collab_interface, abort_signal, fl_ctx)

        job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        job_clients = job_meta.get(JobMetaKey.JOB_CLIENTS)
        all_clients = [from_dict(d) for d in job_clients]
        client_proxies = []
        for c in all_clients:
            p = self._prepare_client_proxy(job_id, cell, c, abort_signal, client_collab_interface, fl_ctx)
            client_proxies.append(p)

        ws = FlareWorkspace(fl_ctx)
        self.client_app.setup(ws, server_proxy, client_proxies, abort_signal)

        self.client_ctx = self.client_app.new_context(self.client_app.name, self.client_app.name)
        self.logger.debug(f"initializing client app {self.client_app.name}")
        self.client_app.initialize(self.client_ctx)

        reply = make_reply(ReturnCode.OK)
        reply[SyncKey.COLLAB_INTERFACE] = client_collab_interface
        return reply

    def _detect_client_class(self) -> Optional[str]:
        """Detect the client class name for class-based clients.

        Returns:
            The class name if using a class-based client (e.g., "FoxClientAPI"),
            None for module-based clients.
        """
        from nvflare.fox.api.module_wrapper import ModuleWrapper

        # Get the underlying client object from ClientApp
        client_obj = self.client_app.obj

        # ModuleWrapper means module-based, no class name needed
        if isinstance(client_obj, ModuleWrapper):
            return None

        # Get class name for class-based clients
        return client_obj.__class__.__name__

    def _start_subprocess_worker(self, cell, client_name: str):
        """Start the subprocess worker for distributed training."""
        self.logger.info(f"Starting subprocess worker for {client_name}...")
        self.logger.info(f"  Training module: {self.training_module}")
        if self.run_cmd:
            self.logger.info(f"  Run command: {self.run_cmd}")

        # Detect client class for Client API mode (e.g., "FoxClientAPI")
        client_class = self._detect_client_class()
        if client_class:
            self.logger.info(f"  Client class: {client_class}")

        self._subprocess_launcher = SubprocessLauncher(
            site_name=client_name,
            training_module=self.training_module,
            parent_cell=cell,
            run_cmd=self.run_cmd,
            subprocess_timeout=self.subprocess_timeout,
            client_class=client_class,
        )

        if not self._subprocess_launcher.start():
            raise RuntimeError("Failed to start subprocess worker")

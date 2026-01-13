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
import inspect
import sys
from types import ModuleType
from typing import Dict, List, Optional

from nvflare.fox.api.app import App, ClientApp, ServerApp
from nvflare.fox.api.filter import FilterChain
from nvflare.fox.api.module_wrapper import ModuleWrapper
from nvflare.fuel.utils.validation_utils import check_positive_int, check_positive_number, check_str
from nvflare.job_config.api import FedJob
from nvflare.recipe.spec import ExecEnv, Recipe

from .controller import FoxController
from .executor import FoxExecutor


def _get_caller_module(stack_level: int = 2) -> Optional[ModuleType]:
    """Get the module of the caller.

    Args:
        stack_level: How many frames to go back (2 = caller of the function that called this)

    Returns:
        The caller's module
    """
    frame = inspect.currentframe()
    try:
        # Go back stack_level frames
        for _ in range(stack_level):
            if frame is not None:
                frame = frame.f_back
        if frame is not None:
            module_name = frame.f_globals.get("__name__", "__main__")
            return sys.modules.get(module_name)
    finally:
        del frame
    return None


def _wrap_if_module(obj):
    """Wrap a module in ModuleWrapper if needed.

    This allows users to pass either:
    - A class instance (used as-is)
    - A Python module containing @fox.collab/@fox.algo functions (auto-wrapped)

    Args:
        obj: Either a class instance or a Python module

    Returns:
        The original object or a ModuleWrapper if it was a module
    """
    if isinstance(obj, ModuleType):
        return ModuleWrapper(obj)
    return obj


class FoxRecipe(Recipe):

    def __init__(
        self,
        job_name: str,
        server: Optional[object] = None,
        client: Optional[object] = None,
        server_objects: Optional[Dict[str, object]] = None,
        client_objects: Optional[Dict[str, object]] = None,
        sync_task_timeout=5,
        max_call_threads_for_server=100,
        max_call_threads_for_client=100,
        min_clients: int = 1,
        # Subprocess execution options (for distributed training like torchrun)
        inprocess: bool = True,
        run_cmd: Optional[str] = None,
        training_module: Optional[str] = None,
        subprocess_timeout: float = 300.0,
    ):
        """Create a FoxRecipe for federated learning.

        Args:
            job_name: Name of the job.
            server: Server object with @fox.algo methods. If None, uses the caller's module.
            client: Client object with @fox.collab methods. If None, uses the caller's module.
            server_objects: Additional named objects for the server.
            client_objects: Additional named objects for clients.
            sync_task_timeout: Timeout for synchronous tasks.
            max_call_threads_for_server: Max threads for server calls.
            max_call_threads_for_client: Max threads for client calls.
            min_clients: Minimum number of clients required.
            inprocess: If True (default), execute training in-process.
                If False, spawn subprocess for training (e.g., for torchrun DDP).
            run_cmd: Command to run the training subprocess
                (e.g., "torchrun --nproc_per_node=4").
            training_module: Python module containing @fox.collab methods.
                Auto-detected from client object in most cases. Only needed
                if auto-detection fails.
            subprocess_timeout: Timeout for subprocess operations.

        Examples:
            # Class-based (traditional)
            recipe = FoxRecipe(job_name="job", server=FedAvg(), client=Trainer())

            # Module-based (pass module directly)
            import my_module
            recipe = FoxRecipe(job_name="job", server=my_module, client=my_module)

            # Simplest: use caller's module (contains both @fox.algo and @fox.collab)
            recipe = FoxRecipe(job_name="job", min_clients=5)

            # Subprocess mode for multi-GPU DDP training (training_module auto-detected)
            recipe = FoxRecipe(
                job_name="fedavg_ddp",
                min_clients=2,
                inprocess=False,
                run_cmd="torchrun --nproc_per_node=4",
            )
        """
        check_str("job_name", job_name)
        check_positive_number("sync_task_timeout", sync_task_timeout)
        check_positive_int("max_call_threads_for_server", max_call_threads_for_server)
        check_positive_int("max_call_threads_for_client", max_call_threads_for_client)
        check_positive_int("min_clients", min_clients)

        # Auto-detect caller's module if server/client not provided
        if server is None or client is None:
            caller_module = _get_caller_module(stack_level=2)
            if server is None:
                server = caller_module
            if client is None:
                client = caller_module

        self.job_name = job_name
        # Auto-wrap modules with ModuleWrapper
        self.server = _wrap_if_module(server)
        self.client = _wrap_if_module(client)
        # Also wrap modules in server_objects/client_objects
        self.server_objects = {k: _wrap_if_module(v) for k, v in server_objects.items()} if server_objects else None
        self.client_objects = {k: _wrap_if_module(v) for k, v in client_objects.items()} if client_objects else None
        self.server_app = ServerApp(self.server)
        self.client_app = ClientApp(self.client)

        if self.server_objects:
            for name, obj in self.server_objects.items():
                self.server_app.add_collab_object(name, obj)

        if self.client_objects:
            for name, obj in self.client_objects.items():
                self.client_app.add_collab_object(name, obj)

        self.sync_task_timeout = sync_task_timeout
        self.max_call_threads_for_server = max_call_threads_for_server
        self.max_call_threads_for_client = max_call_threads_for_client
        self.min_clients = min_clients

        # Subprocess execution options
        self.inprocess = inprocess
        self.run_cmd = run_cmd
        self.subprocess_timeout = subprocess_timeout

        # Auto-detect training_module from client if not provided
        self.training_module = self._detect_training_module(training_module, self.client)

        job = FedJob(name=self.job_name, min_clients=self.min_clients)
        Recipe.__init__(self, job)

    def _detect_training_module(self, explicit_module: Optional[str], client_obj) -> Optional[str]:
        """Auto-detect the training module from client object.

        Handles all cases:
        - Module-based client (ModuleWrapper): get module_name
        - Class-based client: get __class__.__module__
        - Explicit training_module provided: use it directly

        Args:
            explicit_module: Explicitly provided training_module (if any)
            client_obj: The client object (ModuleWrapper or class instance)

        Returns:
            The detected module name, or None if not needed (inprocess=True)
        """
        # If explicitly provided, use it
        if explicit_module:
            return explicit_module

        # If in-process, no training module needed
        if self.inprocess:
            return None

        # Try to detect from client object
        if isinstance(client_obj, ModuleWrapper):
            # Module-based: get from ModuleWrapper
            return client_obj.module_name
        elif hasattr(client_obj, '__class__') and hasattr(client_obj.__class__, '__module__'):
            # Class-based: get from class's module
            module_name = client_obj.__class__.__module__
            # Skip built-in modules
            if module_name and not module_name.startswith('builtins'):
                return module_name

        # Could not auto-detect
        raise ValueError(
            "Could not auto-detect training_module for subprocess execution. "
            "Please provide training_module explicitly."
        )

    def process_env(self, env: ExecEnv):
        """Pass server/client objects to the execution environment.

        This allows SimEnv and FoxPocEnv to receive the server/client objects
        from the recipe without requiring them to be passed twice.
        """
        # Import here to avoid circular dependency
        from nvflare.fox.sim.sim_env import SimEnv
        from nvflare.fox.sys.poc_env import PocEnv

        if isinstance(env, (SimEnv, PocEnv)):
            # Pass server/client objects to the environment if not already set
            if env.server is None:
                env.server = self.server
            if env.client is None:
                env.client = self.client
            if env.server_objects is None:
                env.server_objects = self.server_objects
            if env.client_objects is None:
                env.client_objects = self.client_objects

    def set_server_prop(self, name: str, value):
        self.server_app.set_prop(name, value)

    def set_server_resource_dirs(self, resource_dirs):
        self.server_app.set_resource_dirs(resource_dirs)

    def add_server_outgoing_call_filters(self, pattern: str, filters: List[object]):
        self.server_app.add_outgoing_call_filters(pattern, filters)

    def add_server_incoming_call_filters(self, pattern: str, filters: List[object]):
        self.server_app.add_incoming_call_filters(pattern, filters)

    def add_server_outgoing_result_filters(self, pattern: str, filters: List[object]):
        self.server_app.add_outgoing_result_filters(pattern, filters)

    def add_server_incoming_result_filters(self, pattern: str, filters: List[object]):
        self.server_app.add_incoming_result_filters(pattern, filters)

    def add_client_outgoing_call_filters(self, pattern: str, filters: List[object]):
        self.client_app.add_outgoing_call_filters(pattern, filters)

    def add_client_incoming_call_filters(self, pattern: str, filters: List[object]):
        self.client_app.add_incoming_call_filters(pattern, filters)

    def add_client_outgoing_result_filters(self, pattern: str, filters: List[object]):
        self.client_app.add_outgoing_result_filters(pattern, filters)

    def add_client_incoming_result_filters(self, pattern: str, filters: List[object]):
        self.client_app.add_incoming_result_filters(pattern, filters)

    def set_client_prop(self, name: str, value):
        self.client_app.set_prop(name, value)

    def set_client_resource_dirs(self, resource_dirs):
        self.client_app.set_resource_dirs(resource_dirs)

    def finalize(self) -> FedJob:
        print(f"FoxRecipe: Finalizing job '{self.job_name}'...")
        print("  → Configuring server components...")

        server_obj_id = self.job.to_server(self.server_app.obj, "_server")
        job = self.job

        collab_obj_ids, in_cf_arg, out_cf_arg, in_rf_arg, out_rf_arg = self._create_app_args(
            self.server_app, job.to_server
        )

        controller = FoxController(
            server_obj_id=server_obj_id,
            collab_obj_ids=collab_obj_ids,
            incoming_call_filters=in_cf_arg,
            outgoing_call_filters=out_cf_arg,
            incoming_result_filters=in_rf_arg,
            outgoing_result_filters=out_rf_arg,
            sync_task_timeout=self.sync_task_timeout,
            max_call_threads=self.max_call_threads_for_server,
            props=self.server_app.get_props(),
            resource_dirs=self.server_app.get_resource_dirs(),
        )

        job.to_server(controller, id="controller")

        print("  → Configuring client components...")

        # add client config
        client_obj_id = job.to_clients(self.client_app.obj, "_client")
        c_collab_obj_ids, c_in_cf_arg, c_out_cf_arg, c_in_rf_arg, c_out_rf_arg = self._create_app_args(
            self.client_app, job.to_clients
        )
        executor = FoxExecutor(
            client_obj_id=client_obj_id,
            collab_obj_ids=c_collab_obj_ids,
            incoming_call_filters=c_in_cf_arg,
            outgoing_call_filters=c_out_cf_arg,
            incoming_result_filters=c_in_rf_arg,
            outgoing_result_filters=c_out_rf_arg,
            max_call_threads=self.max_call_threads_for_client,
            props=self.client_app.get_props(),
            resource_dirs=self.client_app.get_resource_dirs(),
            # Subprocess execution options
            inprocess=self.inprocess,
            run_cmd=self.run_cmd,
            training_module=self.training_module,
            subprocess_timeout=self.subprocess_timeout,
        )
        job.to_clients(executor, id="executor", tasks=["*"])

        print("  → Job finalized successfully")
        return job

    def _create_app_args(self, app: App, to_f):
        # collab objs
        collab_obj_ids = []
        collab_objs = app.get_collab_objects()
        for name, obj in collab_objs.items():
            if obj == app.obj:
                # do not include in collab objs since it's done separately.
                continue
            comp_id = to_f(obj, id=name)
            collab_obj_ids.append(comp_id)

        # build filter components
        # since a filter object could be used multiple times, we must make sure that only one component is created
        # for the same object!
        filter_comp_table = {}
        incoming_call_filters = app.get_incoming_call_filters()
        outgoing_call_filters = app.get_outgoing_call_filters()
        incoming_result_filters = app.get_incoming_result_filters()
        outgoing_result_filters = app.get_outgoing_result_filters()

        self._create_filter_components(to_f, incoming_call_filters, filter_comp_table)
        self._create_filter_components(to_f, outgoing_call_filters, filter_comp_table)
        self._create_filter_components(to_f, incoming_result_filters, filter_comp_table)
        self._create_filter_components(to_f, outgoing_result_filters, filter_comp_table)

        # filters
        in_cf_arg = self._create_filter_chain_arg(incoming_call_filters, filter_comp_table)
        out_cf_arg = self._create_filter_chain_arg(outgoing_call_filters, filter_comp_table)
        in_rf_arg = self._create_filter_chain_arg(incoming_result_filters, filter_comp_table)
        out_rf_arg = self._create_filter_chain_arg(outgoing_result_filters, filter_comp_table)
        return collab_obj_ids, in_cf_arg, out_cf_arg, in_rf_arg, out_rf_arg

    @staticmethod
    def _create_filter_chain_arg(filter_chains: list, comp_table: dict):
        result = []
        for chain in filter_chains:
            assert isinstance(chain, FilterChain)
            filter_ids = []
            for f in chain.filters:
                f = f.get_impl_object()
                comp_id = comp_table[id(f)]
                filter_ids.append(comp_id)
            d = {"pattern": chain.pattern, "filters": filter_ids}
            result.append(d)
        return result

    @staticmethod
    def _create_filter_components(to_f, filter_chains: list, comp_table: dict):
        for chain in filter_chains:
            assert isinstance(chain, FilterChain)
            for f in chain.filters:
                f = f.get_impl_object()
                fid = id(f)
                comp_id = comp_table.get(fid)
                if not comp_id:
                    comp_id = to_f(f, id="_filter")
                    comp_table[fid] = comp_id

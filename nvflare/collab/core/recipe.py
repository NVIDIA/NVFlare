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
import importlib
import inspect
import os
from typing import Dict, List, Optional

from nvflare.collab.api.app import App, ClientApp, ServerApp
from nvflare.collab.api.constants import PER_SITE_CONFIG_PROP
from nvflare.collab.api.filter import FilterChain
from nvflare.collab.api.module_wrapper import ModuleWrapper, resolve_server_client, wrap_if_module
from nvflare.collab.runtime.flare.controller import CollabController
from nvflare.collab.runtime.flare.executor import CollabExecutor
from nvflare.fuel.utils.validation_utils import check_positive_int, check_positive_number, check_str
from nvflare.job_config.api import FedJob
from nvflare.recipe.spec import Recipe


class CollabRecipe(Recipe):

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
    ):
        """Create a recipe for collaborative training."""
        check_str("job_name", job_name)
        check_positive_number("sync_task_timeout", sync_task_timeout)
        check_positive_int("max_call_threads_for_server", max_call_threads_for_server)
        check_positive_int("max_call_threads_for_client", max_call_threads_for_client)
        check_positive_int("min_clients", min_clients)

        # When server/client are not specified, use the caller's module
        # (@collab.main / @collab.publish functions defined at module level).
        # Raw modules are wrapped with ModuleWrapper to make them callable and
        # serializable (only the importable module name is stored).
        server, client = resolve_server_client(server, client)

        self.job_name = job_name
        self.server = wrap_if_module(server)
        self.client = wrap_if_module(client)
        self.server_objects = {k: wrap_if_module(v) for k, v in server_objects.items()} if server_objects else None
        self.client_objects = {k: wrap_if_module(v) for k, v in client_objects.items()} if client_objects else None
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
        job = FedJob(name=self.job_name, min_clients=self.min_clients)
        self._finalized = False
        self._per_site_config: Dict[str, dict] = {}
        Recipe.__init__(self, job)

    def _apply_per_site_config(self, config: Dict[str, Dict]) -> None:
        """Deliver per-site config values as per-site client app properties.

        The values become app props for the matching site only, readable in
        client code via ``collab.get_app_prop(name)``. ``CollabExecutor``
        applies each site's entries at start-run in every standard execution
        environment.
        """
        self._per_site_config = {site: dict(values) for site, values in config.items()}

    def set_server_prop(self, name: str, value):
        self.server_app.set_prop(name, value)

    def set_server_resource_dirs(self, resource_dirs):
        self.server_app.set_resource_dirs(resource_dirs)

    def set_client_prop(self, name: str, value):
        self.client_app.set_prop(name, value)

    def set_client_resource_dirs(self, resource_dirs):
        self.client_app.set_resource_dirs(resource_dirs)

    def finalize(self) -> FedJob:
        # finalize() is invoked by both Recipe.run() and Recipe.export(); a recipe
        # instance may go through both (or either twice), so adding the job
        # components must only happen once.
        if self._finalized:
            return self._job
        self._finalized = True

        server_obj_id = self._job.to_server(self.server_app.obj, "_server")
        job = self._job

        collab_obj_ids, in_cf_arg, out_cf_arg, in_rf_arg, out_rf_arg = self._create_app_args(
            self.server_app, job.to_server
        )

        controller = CollabController(
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

        # add client config
        client_obj_id = job.to_clients(self.client_app.obj, "_client")
        c_collab_obj_ids, c_in_cf_arg, c_out_cf_arg, c_in_rf_arg, c_out_rf_arg = self._create_app_args(
            self.client_app, job.to_clients
        )
        executor = CollabExecutor(
            client_obj_id=client_obj_id,
            collab_obj_ids=c_collab_obj_ids,
            incoming_call_filters=c_in_cf_arg,
            outgoing_call_filters=c_out_cf_arg,
            incoming_result_filters=c_in_rf_arg,
            outgoing_result_filters=c_out_rf_arg,
            max_call_threads=self.max_call_threads_for_client,
            props=self._client_props_with_per_site_config(),
            resource_dirs=self.client_app.get_resource_dirs(),
        )
        job.to_clients(executor, id="executor", tasks=["*"])

        # Ship the user-provided server/client code into each app's "custom" folder.
        # A collab job runs user-defined objects (the server/client and any collab
        # objects), so it is inherently bring-your-own-code (BYOC): the code must
        # travel with the job, and the job is authorized via the BYOC right rather
        # than the site class allow-list (which cannot enumerate arbitrary user
        # classes). Adding a custom folder is what marks the job as BYOC.
        for src in self._user_source_files([self.server_app.obj] + list((self.server_objects or {}).values())):
            job.add_file_to_server(src, app_folder_type="custom")
        for src in self._user_source_files([self.client_app.obj] + list((self.client_objects or {}).values())):
            job.add_file_to_clients(src, app_folder_type="custom")
        return job

    def _client_props_with_per_site_config(self) -> Dict[str, object]:
        """Client app props for the executor, with per-site config attached.

        The per-site map rides in the shared executor config under a reserved
        key; each CollabExecutor resolves its own site's entries at start-run.
        """
        props = dict(self.client_app.get_props() or {})
        if self._per_site_config:
            props[PER_SITE_CONFIG_PROP] = self._per_site_config
        return props

    @staticmethod
    def _user_source_files(objs) -> List[str]:
        """Resolve the source .py files backing the given user objects (deduplicated).

        Objects whose source cannot be located (e.g. built-ins or C extensions) are
        skipped.
        """
        files = []
        for obj in objs:
            if obj is None:
                continue
            try:
                if isinstance(obj, ModuleWrapper):
                    target = importlib.import_module(obj.module_name)
                else:
                    target = type(obj)
                src = inspect.getfile(target)
            except (TypeError, OSError, ImportError, ValueError):
                continue
            if src and os.path.isfile(src) and src not in files:
                files.append(src)
        return files

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

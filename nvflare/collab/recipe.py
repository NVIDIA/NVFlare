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
from pathlib import Path
from typing import Dict, List, Optional

from nvflare.apis.job_def import ALL_SITES
from nvflare.collab.api.app import App, ClientApp, ServerApp
from nvflare.collab.api.module_wrapper import ModuleWrapper, resolve_server_client, wrap_if_module
from nvflare.collab.runtime.controller import CollabController
from nvflare.collab.runtime.executor import CollabExecutor
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
        sync_task_timeout=60,
        max_call_threads_for_server=100,
        max_call_threads_for_client=100,
        min_clients: int = 1,
        max_inbound_call_threads_for_server=None,
        max_outbound_call_threads_for_server=None,
        max_inbound_call_threads_for_client=None,
        max_outbound_call_threads_for_client=None,
    ):
        """Create a recipe for collaborative training."""
        check_str("job_name", job_name)
        check_positive_number("sync_task_timeout", sync_task_timeout)
        check_positive_int("max_call_threads_for_server", max_call_threads_for_server)
        check_positive_int("max_call_threads_for_client", max_call_threads_for_client)
        check_positive_int("min_clients", min_clients)
        if max_inbound_call_threads_for_server is not None:
            check_positive_int("max_inbound_call_threads_for_server", max_inbound_call_threads_for_server)
        if max_outbound_call_threads_for_server is not None:
            check_positive_int("max_outbound_call_threads_for_server", max_outbound_call_threads_for_server)
        if max_inbound_call_threads_for_client is not None:
            check_positive_int("max_inbound_call_threads_for_client", max_inbound_call_threads_for_client)
        if max_outbound_call_threads_for_client is not None:
            check_positive_int("max_outbound_call_threads_for_client", max_outbound_call_threads_for_client)

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
        self.max_inbound_call_threads_for_server = (
            max_call_threads_for_server
            if max_inbound_call_threads_for_server is None
            else max_inbound_call_threads_for_server
        )
        self.max_outbound_call_threads_for_server = (
            max_call_threads_for_server
            if max_outbound_call_threads_for_server is None
            else max_outbound_call_threads_for_server
        )
        self.max_inbound_call_threads_for_client = (
            max_call_threads_for_client
            if max_inbound_call_threads_for_client is None
            else max_inbound_call_threads_for_client
        )
        self.max_outbound_call_threads_for_client = (
            max_call_threads_for_client
            if max_outbound_call_threads_for_client is None
            else max_outbound_call_threads_for_client
        )
        self.min_clients = min_clients
        job = FedJob(name=self.job_name, min_clients=self.min_clients)
        self._finalized = False
        self._per_site_config: Dict[str, dict] = {}
        Recipe.__init__(self, job)

    def _apply_per_site_config(self, config: Dict[str, Dict]) -> None:
        """Deliver per-site config values as per-site client app properties.

        Each site's values are materialized only in that site's client app
        configuration and are readable via ``collab.get_app_prop(name)``.
        """
        self._per_site_config = {site: dict(values) for site, values in config.items()}

    def set_server_prop(self, name: str, value):
        self.server_app.set_prop(name, value)

    def set_client_prop(self, name: str, value):
        self.client_app.set_prop(name, value)

    def finalize(self) -> FedJob:
        # finalize() is invoked by both Recipe.run() and Recipe.export(); a recipe
        # instance may go through both (or either twice), so adding the job
        # components must only happen once.
        if self._finalized:
            return self._job

        server_obj_id = self._job.to_server(self.server_app.obj, "_server")
        job = self._job

        collab_obj_ids = self._add_collab_objects(self.server_app, job.to_server)

        controller = CollabController(
            server_obj_id=server_obj_id,
            collab_obj_ids=collab_obj_ids,
            sync_task_timeout=self.sync_task_timeout,
            max_call_threads=self.max_call_threads_for_server,
            max_inbound_call_threads=self.max_inbound_call_threads_for_server,
            max_outbound_call_threads=self.max_outbound_call_threads_for_server,
            props=self.server_app.get_props(),
        )

        job.to_server(controller, id="controller")

        self._ensure_client_apps_prepared()

        # Ship the user-provided server/client code into each app's "custom" folder.
        # A collab job runs user-defined objects (the server/client and any collab
        # objects), so it is inherently bring-your-own-code (BYOC): the code must
        # travel with the job, and the job is authorized via the BYOC right rather
        # than the site class allow-list (which cannot enumerate arbitrary user
        # classes). Adding a custom folder is what marks the job as BYOC.
        server_sources = [self.server_app.obj] + list((self.server_objects or {}).values())
        for src, dest_dir in self._user_source_entries(server_sources):
            job.add_file_to_server(src, dest_dir=dest_dir, app_folder_type="custom")
        client_sources = [self.client_app.obj] + list((self.client_objects or {}).values())
        for src, dest_dir in self._user_source_entries(client_sources):
            client_targets = self.configured_sites() or [ALL_SITES]
            for target in client_targets:
                job.add_file_to(src, target=target, dest_dir=dest_dir, app_folder_type="custom")
        self._finalized = True
        return job

    def _prepare_client_apps(self) -> None:
        """Create either one shared client app or isolated per-site client apps."""
        client_targets = self.configured_sites() or [ALL_SITES]
        for target in client_targets:
            self._add_client_app(target)

    def _add_client_app(self, target: str) -> None:
        job = self._job
        client_obj_id = job.to(self.client_app.obj, target, id="_client")
        c_collab_obj_ids = self._add_collab_objects(
            self.client_app,
            lambda obj, id: job.to(obj, target, id=id),
        )
        props = dict(self.client_app.get_props() or {})
        props.update(self._per_site_config.get(target, {}))
        executor = CollabExecutor(
            client_obj_id=client_obj_id,
            collab_obj_ids=c_collab_obj_ids,
            max_call_threads=self.max_call_threads_for_client,
            max_inbound_call_threads=self.max_inbound_call_threads_for_client,
            max_outbound_call_threads=self.max_outbound_call_threads_for_client,
            props=props,
        )
        job.to(executor, target, id="executor", tasks=["*"])

    @staticmethod
    def _user_source_entries(objs) -> List[tuple[str, Optional[str]]]:
        """Resolve source paths and custom-folder destinations (deduplicated).

        Objects whose source cannot be located (e.g. built-ins or C extensions) are
        skipped. Module wrappers ship their top-level package's Python sources so
        relative imports and sibling modules remain available after job export.
        """
        entries = []
        for obj in objs:
            if obj is None:
                continue
            try:
                if isinstance(obj, ModuleWrapper):
                    target = importlib.import_module(obj.module_name)
                    src = inspect.getfile(target)
                    source_entries = CollabRecipe._module_source_entries(obj.module_name, src)
                else:
                    target = type(obj)
                    src = inspect.getfile(target)
                    source_entries = [(src, None)]
            except (TypeError, OSError, ImportError, ValueError):
                continue
            for entry in source_entries:
                if entry not in entries:
                    entries.append(entry)
        return entries

    @staticmethod
    def _module_source_entries(module_name: str, src: str) -> List[tuple[str, Optional[str]]]:
        """Return package files and import-preserving custom destinations."""
        if not src or not os.path.isfile(src):
            return []

        parts = module_name.split(".")
        if len(parts) == 1:
            return [(src, None)]

        package_dir = Path(src).resolve().parent
        ascents = len(parts) - 1 if Path(src).name == "__init__.py" else len(parts) - 2
        for _ in range(ascents):
            package_dir = package_dir.parent
        if package_dir.is_dir() and package_dir.name == parts[0]:
            entries = []
            for path in sorted(package_dir.rglob("*")):
                if not path.is_file() or path.suffix not in (".py", ".pyi") or "__pycache__" in path.parts:
                    continue
                relative_parent = path.relative_to(package_dir).parent
                dest = Path(parts[0]) / relative_parent
                entries.append((str(path), dest.as_posix()))
            return entries

        # Unusual import loaders may not map the module name to the filesystem
        # hierarchy. Preserve the old single-file behavior in that case.
        return [(src, None)]

    @staticmethod
    def _add_collab_objects(app: App, to_f):
        collab_obj_ids = []
        collab_objs = app.get_collab_objects()
        for name, obj in collab_objs.items():
            if obj == app.obj:
                # do not include in collab objs since it's done separately.
                continue
            comp_id = to_f(obj, id=name)
            collab_obj_ids.append(comp_id)
        return collab_obj_ids

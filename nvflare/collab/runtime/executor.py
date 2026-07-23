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
from typing import Any, Dict, List

from nvflare.apis.client import Client, from_dict
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.collab.api.app import ClientApp
from nvflare.collab.api.constants import MAKE_CLIENT_APP_METHOD, PER_SITE_CONFIG_PROP
from nvflare.collab.api.proxy import Proxy
from nvflare.collab.runtime.cell_dispatcher import CellDispatcher
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.security.logging import secure_format_exception

from .adaptor import CollabAdaptor
from .defs import SETUP_TASK_NAME, SYNC_TASK_NAME, SyncKey
from .dispatch import prepare_for_remote_call


class CollabExecutor(Executor, CollabAdaptor):
    """Client executor for Collab calls in the site's FLARE process."""

    def __init__(
        self,
        client_obj_id: str,
        collab_obj_ids: List[str] = None,
        props: Dict[str, Any] = None,
        max_call_threads=100,
    ):
        Executor.__init__(self)
        CollabAdaptor.__init__(
            self,
            collab_obj_ids=collab_obj_ids,
            props=props,
        )
        self.client_obj_id = client_obj_id
        self.register_event_handler(EventType.START_RUN, self._handle_start_run)
        self.register_event_handler(EventType.END_RUN, self._handle_end_run)
        self.client_app = None
        self.client_ctx = None
        self.thread_executor = ThreadPoolExecutor(max_workers=max_call_threads, thread_name_prefix="collab_call")

    def _handle_start_run(self, event_type: str, fl_ctx: FLContext):
        tensor_decomposer, ok = optional_import(module="nvflare.app_opt.pt.decomposers", name="TensorDecomposer")
        if ok:
            fobs.register(tensor_decomposer)
        engine = fl_ctx.get_engine()
        client_obj = engine.get_component(self.client_obj_id)
        if not client_obj:
            self.system_panic(f"cannot get client component {self.client_obj_id}", fl_ctx)
            return

        client_name = fl_ctx.get_identity_name()

        app = ClientApp(client_obj)

        # If the app contains "make_client_app" method, call it to make the app instance!
        make_client_app_f = getattr(app, MAKE_CLIENT_APP_METHOD, None)
        if not make_client_app_f:
            make_client_app_f = getattr(app.obj, MAKE_CLIENT_APP_METHOD, None)
        if make_client_app_f and callable(make_client_app_f):
            app = make_client_app_f(client_name)
            if not isinstance(app, ClientApp):
                raise RuntimeError(f"result returned by {MAKE_CLIENT_APP_METHOD} must be ClientApp but got {type(app)}")

        app.name = client_name

        err = self.process_config(app, fl_ctx)
        if err:
            self.system_panic(err, fl_ctx)
            return

        # Resolve this site's entries from the recipe's per-site config into
        # plain app props (readable via collab.get_app_prop).
        per_site = app.get_prop(PER_SITE_CONFIG_PROP)
        if per_site:
            for name, value in (per_site.get(client_name) or {}).items():
                app.set_prop(name, value)

        # Publish the app only after start-run initialization succeeds. This
        # keeps execute() from using a partially configured app.
        self.client_app = app

    def _handle_end_run(self, event_type: str, fl_ctx: FLContext):
        try:
            if self.client_ctx:
                self.logger.info(f"finalizing client app {self.client_app.name}")
                self.client_app.finalize(self.client_ctx)
        finally:
            self.thread_executor.shutdown(wait=True, cancel_futures=True)

    def _prepare_server_proxy(self, server_fqcn, cell, collab_interface: dict, abort_signal, fl_ctx: FLContext):
        server_name = "server"
        backend = CellDispatcher(
            manager=self,
            engine=fl_ctx.get_engine(),
            caller=self.client_app.name,
            cell=cell,
            target_fqcn=server_fqcn,
            abort_signal=abort_signal,
            thread_executor=self.thread_executor,
        )
        proxy = Proxy(
            app=self.client_app,
            target_name=server_name,
            target_fqn=server_name,
            backend=backend,
            target_interface=collab_interface.get(""),
        )

        for name, itf in collab_interface.items():
            if name == "":
                # this is the server app itself
                continue
            p = Proxy(
                app=self.client_app,
                target_name=f"{server_name}.{name}",
                target_fqn="",
                backend=backend,
                target_interface=itf,
            )
            proxy.add_child(name, p)
        return proxy

    def _prepare_client_proxy(self, job_id, cell, client: Client, abort_signal, collab_interface, fl_ctx: FLContext):
        backend = CellDispatcher(
            manager=self,
            engine=fl_ctx.get_engine(),
            caller=self.client_app.name,
            cell=cell,
            target_fqcn=FQCN.join([client.get_fqcn(), job_id]),
            abort_signal=abort_signal,
            thread_executor=self.thread_executor,
        )
        proxy = Proxy(
            app=self.client_app,
            target_name=client.name,
            target_fqn=client.get_fqsn(),
            backend=backend,
            target_interface=collab_interface.get(""),
        )

        collab_objs = self.client_app.get_collab_objects()
        if collab_objs:
            for name in collab_objs.keys():
                p = Proxy(
                    app=self.client_app,
                    target_name=f"{client.name}.{name}",
                    target_fqn="",
                    backend=backend,
                    target_interface=collab_interface.get(name),
                )
                proxy.add_child(name, p)
        return proxy

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name not in (SYNC_TASK_NAME, SETUP_TASK_NAME):
            self.log_error(fl_ctx, f"received unsupported task {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)

        if self.client_app is None:
            self.log_error(fl_ctx, "client app is unavailable because start-run initialization did not complete")
            return make_reply(ReturnCode.ERROR)

        client_collab_interface = self.client_app.get_collab_interface()
        if task_name == SYNC_TASK_NAME:
            reply = make_reply(ReturnCode.OK)
            reply[SyncKey.COLLAB_INTERFACE] = client_collab_interface
            return reply

        server_collab_interface = shareable.get(SyncKey.COLLAB_INTERFACE)
        server_fqcn = shareable.get(SyncKey.SERVER_FQCN)
        client_interfaces = shareable.get(SyncKey.CLIENT_INTERFACES)
        if not isinstance(server_fqcn, str) or not server_fqcn:
            self.log_error(fl_ctx, "missing server FQCN in setup task")
            return make_reply(ReturnCode.BAD_TASK_DATA)
        if not isinstance(server_collab_interface, dict) or not isinstance(client_interfaces, dict):
            self.log_error(fl_ctx, "missing collab interfaces in setup task")
            return make_reply(ReturnCode.BAD_TASK_DATA)
        self.log_info(fl_ctx, f"{client_collab_interface=} {server_collab_interface=}")

        engine = fl_ctx.get_engine()
        cell = engine.get_cell()
        client_name = fl_ctx.get_identity_name()
        job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        job_clients = job_meta.get(JobMetaKey.JOB_CLIENTS)
        all_clients = [from_dict(d) for d in job_clients]

        prepare_for_remote_call(
            cell=cell,
            app=self.client_app,
            logger=self.logger,
            executor=self.thread_executor,
        )

        # build proxies
        job_id = fl_ctx.get_job_id()
        server_proxy = self._prepare_server_proxy(server_fqcn, cell, server_collab_interface, abort_signal, fl_ctx)

        client_proxies = []
        for c in all_clients:
            remote_interface = client_interfaces.get(c.name)
            if not isinstance(remote_interface, dict):
                self.log_error(fl_ctx, f"missing collab interface for client {c.name}")
                return make_reply(ReturnCode.BAD_TASK_DATA)
            p = self._prepare_client_proxy(job_id, cell, c, abort_signal, remote_interface, fl_ctx)
            client_proxies.append(p)

        ws = fl_ctx.get_workspace()
        try:
            self.client_app.setup(ws, server_proxy, client_proxies, abort_signal)
        except Exception as ex:
            self.client_ctx = None
            self.log_exception(
                fl_ctx,
                f"failed to set up client app {self.client_app.name}: {secure_format_exception(ex)}",
            )
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        client_ctx = self.client_app.new_context(self.client_app.name, self.client_app.name, set_call_ctx=False)
        self.logger.info(f"initializing client app {self.client_app.name}")
        try:
            self.client_app.initialize(client_ctx)
        except Exception as ex:
            self.client_ctx = None
            self.log_exception(
                fl_ctx,
                f"failed to initialize client app {self.client_app.name}: {secure_format_exception(ex)}",
            )
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        self.client_ctx = client_ctx

        return make_reply(ReturnCode.OK)

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
from nvflare.collab.api.constants import MAKE_CLIENT_APP_METHOD, BackendType
from nvflare.collab.api.proxy import Proxy
from nvflare.fuel.f3.cellnet.fqcn import FQCN

from .adaptor import CollabAdaptor
from .backend import FlareBackend
from .constants import SYNC_TASK_NAME, SyncKey
from .utils import prepare_for_remote_call
from .ws import FlareWorkspace


class CollabExecutor(Executor, CollabAdaptor):

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
    ):
        Executor.__init__(self)
        CollabAdaptor.__init__(
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
        self.thread_executor = ThreadPoolExecutor(max_workers=max_call_threads, thread_name_prefix="collab_call")

    def _handle_start_run(self, event_type: str, fl_ctx: FLContext):
        fl_ctx.set_prop(FLContextKey.COLLAB_MODE, True, private=True, sticky=True)
        engine = fl_ctx.get_engine()
        client_obj = engine.get_component(self.client_obj_id)
        if not client_obj:
            self.system_panic(f"cannot get client component {self.client_obj_id}", fl_ctx)
            return

        client_name = fl_ctx.get_identity_name()

        app = ClientApp(client_obj)

        # If the app contains "make_client_app" method, call it to make the app instance!
        make_client_app_f = getattr(app, MAKE_CLIENT_APP_METHOD, None)
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
        if self.client_ctx:
            self.logger.info(f"finalizing client app {self.client_app.name}")
            self.client_app.finalize(self.client_ctx)
        self.thread_executor.shutdown(wait=True, cancel_futures=True)

    def _prepare_server_proxy(self, job_id, cell, collab_interface: dict, abort_signal, fl_ctx: FLContext):
        server_name = "server"
        backend = FlareBackend(
            manager=self,
            engine=fl_ctx.get_engine(),
            caller=self.client_app.name,
            cell=cell,
            target_fqcn=FQCN.join([FQCN.ROOT_SERVER, job_id]),
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
        backend = FlareBackend(
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
        if task_name != SYNC_TASK_NAME:
            self.log_error(fl_ctx, f"received unsupported task {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)

        server_collab_interface = shareable.get(SyncKey.COLLAB_INTERFACE)
        client_collab_interface = self.client_app.get_collab_interface()
        self.log_info(fl_ctx, f"{client_collab_interface=} {server_collab_interface=}")

        engine = fl_ctx.get_engine()
        cell = engine.get_cell()

        prepare_for_remote_call(
            cell=cell,
            app=self.client_app,
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
        self.logger.info(f"initializing client app {self.client_app.name}")
        self.client_app.initialize(self.client_ctx)

        reply = make_reply(ReturnCode.OK)
        reply[SyncKey.COLLAB_INTERFACE] = client_collab_interface
        return reply

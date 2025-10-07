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
from typing import Dict

from nvflare.apis.client import Client, from_dict
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.fox.api.app import ClientApp, ClientAppFactory
from nvflare.fox.api.constants import EnvType
from nvflare.fox.api.proxy import Proxy
from nvflare.fuel.f3.cellnet.fqcn import FQCN

from .adaptor import FoxAdaptor
from .backend import SysBackend
from .constants import SYNC_TASK_NAME, SyncKey
from .utils import prepare_for_remote_call


class FoxExecutor(Executor, FoxAdaptor):

    def __init__(
        self,
        client_app_id: str,
        collab_obj_ids: Dict[str, str] = None,
        incoming_call_filters=None,
        outgoing_call_filters=None,
        incoming_result_filters=None,
        outgoing_result_filters=None,
        max_call_threads=100,
    ):
        Executor.__init__(self)
        FoxAdaptor.__init__(
            self,
            collab_obj_ids=collab_obj_ids,
            incoming_call_filters=incoming_call_filters,
            outgoing_call_filters=outgoing_call_filters,
            incoming_result_filters=incoming_result_filters,
            outgoing_result_filters=outgoing_result_filters,
        )
        self.client_app_id = client_app_id
        self.register_event_handler(EventType.START_RUN, self._handle_start_run)
        self.register_event_handler(EventType.END_RUN, self._handle_end_run)
        self.client_app = None
        self.thread_executor = ThreadPoolExecutor(max_workers=max_call_threads)

    def _handle_start_run(self, event_type: str, fl_ctx: FLContext):
        fl_ctx.set_prop(FLContextKey.FOX_MODE, True, private=True, sticky=True)
        engine = fl_ctx.get_engine()
        client_app = engine.get_component(self.client_app_id)
        if not isinstance(client_app, (ClientApp, ClientAppFactory)):
            self.system_panic(
                f"component {self.client_app_id} must be ClientApp or ClientAppFactory but got {type(client_app)}",
                fl_ctx,
            )
            return

        client_name = fl_ctx.get_identity_name()
        if isinstance(client_app, ClientApp):
            self.client_app = client_app
        else:
            self.client_app = client_app.make_client_app(client_name)

        self.client_app.name = client_name
        self.client_app.env_type = EnvType.SYSTEM

        err = self.process_config(self.client_app, fl_ctx)
        if err:
            self.system_panic(err, fl_ctx)

    def _handle_end_run(self, event_type: str, fl_ctx: FLContext):
        self.thread_executor.shutdown(wait=False, cancel_futures=True)

    def _prepare_server_proxy(self, job_id, cell, collab_interface: dict, abort_signal):
        server_name = "server"
        backend = SysBackend(
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

    def _prepare_client_proxy(self, job_id, cell, client: Client, abort_signal, collab_interface):
        backend = SysBackend(
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

        if self.collab_obj_ids:
            for name in self.collab_obj_ids.keys():
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
        server_proxy = self._prepare_server_proxy(job_id, cell, server_collab_interface, abort_signal)

        job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        job_clients = job_meta.get(JobMetaKey.JOB_CLIENTS)
        all_clients = [from_dict(d) for d in job_clients]
        client_proxies = []
        for c in all_clients:
            p = self._prepare_client_proxy(job_id, cell, c, abort_signal, client_collab_interface)
            client_proxies.append(p)

        self.client_app.setup(server_proxy, client_proxies, abort_signal)

        ctx = self.client_app.new_context(self.client_app.name, self.client_app.name)
        self.client_app.initialize(ctx)

        reply = make_reply(ReturnCode.OK)
        reply[SyncKey.COLLAB_INTERFACE] = client_collab_interface
        return reply

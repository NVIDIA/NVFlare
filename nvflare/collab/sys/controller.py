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
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from nvflare.apis.client import Client as ClientSite
from nvflare.apis.controller_spec import Client, ClientTask, Task
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal
from nvflare.collab.api.app import ServerApp
from nvflare.collab.api.constants import BackendType
from nvflare.collab.api.proxy import Proxy
from nvflare.collab.api.run_server import run_server
from nvflare.fuel.f3.cellnet.fqcn import FQCN

from .adaptor import CollabAdaptor
from .backend import FlareBackend
from .constants import SYNC_TASK_NAME, SyncKey
from .utils import prepare_for_remote_call
from .ws import FlareWorkspace


class _ClientInfo:

    def __init__(self, collab_interface: dict):
        """Information about a client. Reported by the client in the sync response.

        Args:
            collab_interface: collab method interface of the client.
        """
        self.publish_interface = collab_interface


class CollabController(Controller, CollabAdaptor):

    def __init__(
        self,
        server_obj_id: str = None,
        collab_obj_ids: List[str] = None,
        incoming_call_filters=None,
        outgoing_call_filters=None,
        incoming_result_filters=None,
        outgoing_result_filters=None,
        props=None,
        resource_dirs=None,
        sync_task_timeout=5,
        max_call_threads=100,
    ):
        Controller.__init__(self)
        CollabAdaptor.__init__(
            self,
            props=props,
            resource_dirs=resource_dirs,
            collab_obj_ids=collab_obj_ids,
            incoming_call_filters=incoming_call_filters,
            outgoing_call_filters=outgoing_call_filters,
            incoming_result_filters=incoming_result_filters,
            outgoing_result_filters=outgoing_result_filters,
        )
        self.server_obj_id = server_obj_id  # component name
        self.sync_task_timeout = sync_task_timeout
        self.server_app = None
        self.client_info = {}  # client name => _ClientInfo
        self.cell = None
        self.thread_executor = ThreadPoolExecutor(max_workers=max_call_threads, thread_name_prefix="fox_call")

    def start_controller(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()

        server_obj = engine.get_component(self.server_obj_id)
        if not server_obj:
            self.system_panic(f"no component defined for {self.server_obj_id}", fl_ctx)
            return

        app = ServerApp(server_obj)

        app.name = "server"
        app.set_backend_type(BackendType.FLARE)

        err = self.process_config(app, fl_ctx)
        if err:
            self.system_panic(err, fl_ctx)
            return
        self.server_app = app

    def _prepare_client_backend(self, job_id, client: ClientSite, abort_signal: Signal, fl_ctx: FLContext):
        return FlareBackend(
            manager=self,
            engine=fl_ctx.get_engine(),
            caller=self.server_app.name,
            cell=self.cell,
            target_fqcn=FQCN.join([client.get_fqcn(), job_id]),
            abort_signal=abort_signal,
            thread_executor=self.thread_executor,
        )

    def _prepare_server_backend(self, job_id: str, abort_signal: Signal, fl_ctx: FLContext):
        return FlareBackend(
            manager=self,
            engine=fl_ctx.get_engine(),
            caller=self.server_app.name,
            cell=self.cell,
            target_fqcn=FQCN.join([FQCN.ROOT_SERVER, job_id]),
            abort_signal=abort_signal,
            thread_executor=self.thread_executor,
        )

    def _prepare_client_proxy(
        self,
        job_id: str,
        client: ClientSite,
        collab_interface: dict,
        abort_signal,
        fl_ctx: FLContext,
    ):
        backend = self._prepare_client_backend(job_id, client, abort_signal, fl_ctx)
        proxy = Proxy(
            app=self.server_app,
            target_name=client.name,
            target_fqn=client.get_fqsn(),
            backend=backend,
            target_interface=collab_interface.get(""),
        )

        for name, itf in collab_interface.items():
            if name == "":
                continue

            p = Proxy(
                app=self.server_app,
                target_name=f"{client.name}.{name}",
                target_fqn="",
                backend=backend,
                target_interface=itf,
            )
            proxy.add_child(name, p)
        return proxy

    def _prepare_server_proxy(
        self,
        job_id,
        abort_signal,
        collab_interface: dict,
        fl_ctx: FLContext,
    ):
        server_name = self.server_app.name
        backend = self._prepare_server_backend(job_id, abort_signal, fl_ctx)
        proxy = Proxy(
            app=self.server_app,
            target_name=server_name,
            target_fqn=server_name,
            backend=backend,
            target_interface=collab_interface.get(""),
        )

        collab_objs = self.server_app.get_collab_objects()
        if collab_objs:
            for name in collab_objs.keys():
                p = Proxy(
                    app=self.server_app,
                    target_name=f"{server_name}.{name}",
                    target_fqn="",
                    backend=backend,
                    target_interface=collab_interface.get(name),
                )
                proxy.add_child(name, p)
        return proxy

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        # configure all sites
        server_collab_interface = self.server_app.get_collab_interface()
        task_data = Shareable({SyncKey.COLLAB_INTERFACE: server_collab_interface})
        task = Task(
            name=SYNC_TASK_NAME,
            data=task_data,
            timeout=int(self.sync_task_timeout),
            result_received_cb=self._process_sync_reply,
        )

        engine = fl_ctx.get_engine()
        self.logger.info(f"server engine {type(engine)}")
        all_clients = engine.get_clients()
        num_clients = len(all_clients)
        for c in all_clients:
            self.client_info[c.name] = None

        start_time = time.time()
        self.broadcast_and_wait(
            task=task,
            min_responses=num_clients,
            abort_signal=abort_signal,
            fl_ctx=fl_ctx,
        )
        time_taken = time.time() - start_time
        self.log_info(fl_ctx, f"client sync took {time_taken} seconds")

        failed_clients = []
        for c, info in self.client_info.items():
            if not info:
                failed_clients.append(c)

        if failed_clients:
            self.system_panic(
                f"failed to sync clients {failed_clients}",
                fl_ctx,
            )
            return

        self.log_info(fl_ctx, f"successfully synced clients {self.client_info.keys()}")

        # register msg CB for processing object calls
        self.cell = engine.get_cell()
        prepare_for_remote_call(self.cell, self.server_app, self.logger)

        # prepare proxies and backends
        job_id = fl_ctx.get_job_id()
        server_proxy = self._prepare_server_proxy(job_id, abort_signal, server_collab_interface, fl_ctx)
        client_proxies = []
        for c in all_clients:
            info = self.client_info[c.name]
            # assert isinstance(info, _ClientInfo)
            client_proxies.append(self._prepare_client_proxy(job_id, c, info.publish_interface, abort_signal, fl_ctx))

        ws = FlareWorkspace(fl_ctx)
        self.server_app.setup(ws, server_proxy, client_proxies, abort_signal)
        run_server(self.server_app, self.logger)

    def _process_sync_reply(self, client_task: ClientTask, fl_ctx: FLContext):
        result = client_task.result
        client_name = client_task.client.name

        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            self.log_info(fl_ctx, f"successfully synced client {client_name}")
            collab_itf = result.get(SyncKey.COLLAB_INTERFACE)
            self.client_info[client_name] = _ClientInfo(collab_itf)
        else:
            self.log_error(fl_ctx, f"client {client_task.client.name} failed to sync: {rc}")
            self.client_info[client_name] = None

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        pass

    def stop_controller(self, fl_ctx: FLContext):
        self.thread_executor.shutdown(wait=True, cancel_futures=True)

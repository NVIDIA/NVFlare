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
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from nvflare.apis.client import Client as ClientSite
from nvflare.apis.controller_spec import Client, ClientTask, Task
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal
from nvflare.focs.api.app import ServerApp
from nvflare.focs.api.constants import ContextKey
from nvflare.focs.api.proxy import Proxy
from nvflare.focs.api.strategy import Strategy

from .backend import SysBackend
from .constants import SYNC_TASK_NAME, SyncKey
from .utils import prepare_for_remote_call


class _ClientInfo:

    def __init__(self, target_obj_names):
        """Information about a client. Reported by the client in the sync response.

        Args:
            target_obj_names: names of target objects on the client.
        """
        self.target_obj_names = target_obj_names


class FocsController(Controller):

    def __init__(
        self,
        server_app_id: str,
        strategy_ids: List[str],
        server_target_obj_ids: Dict[str, str] = None,
        sync_task_timeout=2,
        max_call_threads=100,
    ):
        Controller.__init__(self)
        self.server_app_id = server_app_id  # component name
        self.strategy_ids = strategy_ids  # component names
        self.server_target_obj_ids = server_target_obj_ids  # component IDs
        self.sync_task_timeout = sync_task_timeout
        self.server_app = None
        self.client_info = {}  # client name => _ClientInfo
        self.cell = None
        self.thread_executor = ThreadPoolExecutor(max_workers=max_call_threads)

        if not strategy_ids:
            raise ValueError(f"no strategies defined - there must be at least one strategy")

    def start_controller(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        app = engine.get_component(self.server_app_id)
        if not isinstance(app, ServerApp):
            self.system_panic(f"component {self.server_app_id} must be ServerApp but got {type(app)}", fl_ctx)
            return

        app.name = "server"

        for cid in self.strategy_ids:
            strategy = engine.get_component(cid)
            if not isinstance(strategy, Strategy):
                self.system_panic(f"component {cid} must be Strategy but got {type(strategy)}", fl_ctx)
                return

            app.add_strategy(strategy)

        if self.server_target_obj_ids:
            for name, cid in self.server_target_obj_ids:
                obj = engine.get_component(cid)
                if not obj:
                    self.system_panic(f"component {cid} does not exist", fl_ctx)
                    return

                app.add_target_object(name, obj)

        self.server_app = app

        # register msg CB for processing object calls
        self.cell = engine.get_cell()
        prepare_for_remote_call(self.cell, self.server_app, self.logger)

    def _prepare_client_backend(self, client: ClientSite, abort_signal: Signal):
        return SysBackend(
            caller=self.server_app.name,
            cell=self.cell,
            target_fqcn=client.get_fqcn(),
            abort_signal=abort_signal,
            thread_executor=self.thread_executor,
        )

    def _prepare_server_backend(self, abort_signal: Signal):
        return SysBackend(
            caller=self.server_app.name,
            cell=self.cell,
            target_fqcn="server",
            abort_signal=abort_signal,
            thread_executor=self.thread_executor,
        )

    def _prepare_client_proxy(self, client: ClientSite, target_obj_names: List[str], abort_signal):
        backend = self._prepare_client_backend(client, abort_signal)
        proxy = Proxy(app=self.server_app, target_name=client.name, backend=backend, caller_name=self.server_app.name)

        for name in target_obj_names:
            p = Proxy(
                app=self.server_app,
                target_name=f"{client.name}.{name}",
                backend=backend,
                caller_name=self.server_app.name,
            )
            setattr(proxy, name, p)
        return proxy

    def _prepare_server_proxy(self, abort_signal):
        server_name = self.server_app.name
        backend = self._prepare_server_backend(abort_signal)
        proxy = Proxy(app=self.server_app, target_name=server_name, backend=backend, caller_name=server_name)

        tos = self.server_app.get_target_objects()
        if tos:
            for name in tos.keys():
                p = Proxy(
                    app=self.server_app, target_name=f"{server_name}.{name}", backend=backend, caller_name=server_name
                )
                setattr(proxy, name, p)
        return proxy

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        # configure all sites
        if self.server_target_obj_ids:
            target_obj_names = list(self.server_target_obj_ids.keys())
        else:
            target_obj_names = []

        task_data = Shareable({SyncKey.TARGET_OBJ_NAMES: target_obj_names})
        task = Task(
            name=SYNC_TASK_NAME,
            data=task_data,
            timeout=self.sync_task_timeout,
            result_received_cb=self._process_sync_reply,
        )

        engine = fl_ctx.get_engine()
        all_clients = engine.get_clients()
        num_clients = len(all_clients)
        for c in all_clients:
            assert isinstance(c, ClientSite)
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

        # prepare proxies and backends
        server_proxy = self._prepare_server_proxy(abort_signal)
        client_proxies = []
        for c in all_clients:
            info = self.client_info[c.name]
            assert isinstance(info, _ClientInfo)
            client_proxies.append(self._prepare_client_proxy(c, info.target_obj_names, abort_signal))

        self.server_app.setup(server_proxy, client_proxies, abort_signal)

        server_ctx = self.server_app.new_context(caller=self.server_app.name, callee=self.server_app.name)
        self.log_info(fl_ctx, "initializing server app")
        self.server_app.initialize(server_ctx)

        for idx, strategy in enumerate(self.server_app.strategies):
            if abort_signal.triggered:
                break

            try:
                self.log_info(fl_ctx, f"Running Strategy #{idx + 1} - {type(strategy).__name__}")
                self.server_app.current_strategy = strategy
                result = strategy.execute(context=server_ctx)
                server_ctx.set_prop(ContextKey.INPUT, result)
            except:
                traceback.print_exc()
                break

    def _process_sync_reply(self, client_task: ClientTask, fl_ctx: FLContext):
        result = client_task.result
        client_name = client_task.client.name

        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            self.log_info(fl_ctx, f"successfully synced client {client_name}")
            target_obj_names = result.get(SyncKey.TARGET_OBJ_NAMES)
            if not target_obj_names:
                target_obj_names = []
            self.client_info[client_name] = _ClientInfo(target_obj_names)
        else:
            self.log_error(fl_ctx, f"client {client_task.client.name} failed to sync: {rc}")
            self.client_info[client_name] = None

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        pass

    def stop_controller(self, fl_ctx: FLContext):
        self.thread_executor.shutdown(wait=False, cancel_futures=True)

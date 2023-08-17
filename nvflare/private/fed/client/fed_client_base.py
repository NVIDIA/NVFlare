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

import logging
import threading
import time
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from typing import List, Optional

from nvflare.apis.filter import Filter
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, SecureTrainConst, ServerCommandKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import FLCommunicationError
from nvflare.apis.overseer_spec import SP
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.mpm import MainProcessMonitor as mpm
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.private.defs import EngineConstant
from nvflare.security.logging import secure_format_exception

from .client_status import ClientStatus
from .communicator import Communicator


def _check_progress(remote_tasks):
    if remote_tasks[0] is not None:
        # shareable = fobs.loads(remote_tasks[0].payload)
        shareable = remote_tasks[0].payload
        return True, shareable.get_header(ServerCommandKey.TASK_NAME), shareable
    else:
        return False, None, None


class FederatedClientBase:
    """The client-side base implementation of federated learning.

    This class provide the tools function which will be used in both FedClient and FedClientLite.
    """

    def __init__(
        self,
        client_name,
        client_args,
        secure_train,
        server_args=None,
        retry_timeout=30,
        client_state_processors: Optional[List[Filter]] = None,
        handlers: Optional[List[FLComponent]] = None,
        compression=None,
        overseer_agent=None,
        args=None,
        components=None,
        cell: Cell = None,
    ):
        """To init FederatedClientBase.

        Args:
            client_name: client name
            client_args: client config args
            secure_train: True/False to indicate secure train
            server_args: server config args
            retry_timeout: retry timeout
            client_state_processors: client state processor filters
            handlers: handlers
            compression: communication compression algorithm
            cell: CellNet communicator
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.client_name = client_name
        self.token = None
        self.ssid = None
        self.client_args = client_args
        self.servers = server_args
        self.cell = cell
        self.net_agent = None
        self.args = args
        self.engine_create_timeout = client_args.get("engine_create_timeout", 15.0)
        self.cell_check_frequency = client_args.get("cell_check_frequency", 0.005)

        self.communicator = Communicator(
            ssl_args=client_args,
            secure_train=secure_train,
            client_state_processors=client_state_processors,
            compression=compression,
            cell=cell,
            client_register_interval=client_args.get("client_register_interval", 2.0),
            timeout=client_args.get("communication_timeout", 30.0),
        )

        self.secure_train = secure_train
        self.handlers = handlers
        self.components = components

        self.heartbeat_done = False
        self.fl_ctx = FLContext()
        self.platform = None
        self.abort_signal = Signal()
        self.engine = None
        self.client_runner = None

        self.status = ClientStatus.NOT_STARTED
        self.remote_tasks = None

        self.sp_established = False
        self.overseer_agent = overseer_agent

        self.overseer_agent = self._init_agent(args)

        if secure_train:
            if self.overseer_agent:
                self.overseer_agent.set_secure_context(
                    ca_path=client_args["ssl_root_cert"],
                    cert_path=client_args["ssl_cert"],
                    prv_key_path=client_args["ssl_private_key"],
                )

    def start_overseer_agent(self):
        if self.overseer_agent:
            self.overseer_agent.start(self.overseer_callback)

    def _init_agent(self, args=None):
        kv_list = parse_vars(args.set)
        sp = kv_list.get("sp")

        if sp:
            fl_ctx = FLContext()
            fl_ctx.set_prop(FLContextKey.SP_END_POINT, sp)
            self.overseer_agent.initialize(fl_ctx)

        return self.overseer_agent

    def overseer_callback(self, overseer_agent):
        if overseer_agent.is_shutdown():
            self.engine.shutdown()
            return

        sp = overseer_agent.get_primary_sp()
        self.set_primary_sp(sp)

    def set_sp(self, project_name, sp: SP):
        if sp and sp.primary is True:
            server = self.servers[project_name].get("target")
            location = sp.name + ":" + sp.fl_port
            if server != location:
                self.servers[project_name]["target"] = location
                self.sp_established = True

                scheme = self.servers[project_name].get("scheme", "grpc")
                scheme_location = scheme + "://" + location
                if self.cell:
                    self.cell.change_server_root(scheme_location)
                else:
                    self._create_cell(location, scheme)

                self.logger.info(f"Got the new primary SP: {scheme_location}")

            if self.ssid and self.ssid != sp.service_session_id:
                self.ssid = sp.service_session_id
                thread = threading.Thread(target=self._switch_ssid)
                thread.start()

    def _create_cell(self, location, scheme):
        if self.args.job_id:
            fqcn = FQCN.join([self.client_name, self.args.job_id])
            parent_url = self.args.parent_url
            create_internal_listener = False
        else:
            fqcn = self.client_name
            parent_url = None
            create_internal_listener = True
        if self.secure_train:
            root_cert = self.client_args[SecureTrainConst.SSL_ROOT_CERT]
            ssl_cert = self.client_args[SecureTrainConst.SSL_CERT]
            private_key = self.client_args[SecureTrainConst.PRIVATE_KEY]

            credentials = {
                DriverParams.CA_CERT.value: root_cert,
                DriverParams.CLIENT_CERT.value: ssl_cert,
                DriverParams.CLIENT_KEY.value: private_key,
            }
        else:
            credentials = {}
        self.cell = Cell(
            fqcn=fqcn,
            root_url=scheme + "://" + location,
            secure=self.secure_train,
            credentials=credentials,
            create_internal_listener=create_internal_listener,
            parent_url=parent_url,
        )
        self.cell.start()
        self.communicator.cell = self.cell
        self.net_agent = NetAgent(self.cell)
        mpm.add_cleanup_cb(self.net_agent.close)
        mpm.add_cleanup_cb(self.cell.stop)

        if self.args.job_id:
            start = time.time()
            self.logger.info("Wait for client_runner to be created.")
            while not self.client_runner:
                if time.time() - start > self.engine_create_timeout:
                    raise RuntimeError(f"Failed get client_runner after {self.engine_create_timeout} seconds")
                time.sleep(self.cell_check_frequency)
            self.logger.info(f"Got client_runner after {time.time()-start} seconds")
            self.client_runner.engine.cell = self.cell
        else:
            start = time.time()
            self.logger.info("Wait for engine to be created.")
            while not self.engine:
                if time.time() - start > self.engine_create_timeout:
                    raise RuntimeError(f"Failed to get engine after {time.time()-start} seconds")
                time.sleep(self.cell_check_frequency)
            self.logger.info(f"Got engine after {time.time() - start} seconds")
            self.engine.cell = self.cell
            self.engine.admin_agent.register_cell_cb()

    def _switch_ssid(self):
        if self.engine:
            for job_id in self.engine.get_all_job_ids():
                self.engine.abort_task(job_id)
        # self.register()
        self.logger.info(f"Primary SP switched to new SSID: {self.ssid}")

    def client_register(self, project_name):
        """Register the client to the FL server.

        Args:
            project_name: FL study project name.
        """
        if not self.token:
            try:
                self.token, self.ssid = self.communicator.client_registration(
                    self.client_name, self.servers, project_name
                )
                if self.token is not None:
                    self.fl_ctx.set_prop(FLContextKey.CLIENT_NAME, self.client_name, private=False)
                    self.fl_ctx.set_prop(EngineConstant.FL_TOKEN, self.token, private=False)
                    self.logger.info(
                        "Successfully registered client:{} for project {}. Token:{} SSID:{}".format(
                            self.client_name, project_name, self.token, self.ssid
                        )
                    )

            except FLCommunicationError:
                self.communicator.heartbeat_done = True

    def fetch_execute_task(self, project_name, fl_ctx: FLContext):
        """Fetch a task from the server.

        Args:
            project_name: FL study project name
            fl_ctx: FLContext

        Returns:
            A CurrentTask message from server
        """
        try:
            self.logger.debug("Starting to fetch execute task.")
            task = self.communicator.pull_task(self.servers, project_name, self.token, self.ssid, fl_ctx)

            return task
        except FLCommunicationError as e:
            self.logger.info(secure_format_exception(e))

    def push_execute_result(self, project_name, shareable: Shareable, fl_ctx: FLContext):
        """Submit execution results of a task to server.

        Args:
            project_name: FL study project name
            shareable: Shareable object
            fl_ctx: FLContext

        Returns:
            A FederatedSummary message from the server.
        """
        try:
            self.logger.info("Starting to push execute result.")
            execute_task_name = fl_ctx.get_prop(FLContextKey.TASK_NAME)
            return_code = self.communicator.submit_update(
                self.servers,
                project_name,
                self.token,
                self.ssid,
                fl_ctx,
                self.client_name,
                shareable,
                execute_task_name,
            )

            return return_code
        except FLCommunicationError as e:
            self.logger.info(secure_format_exception(e))

    def send_heartbeat(self, project_name, interval):
        try:
            if self.token:
                start = time.time()
                while not self.engine:
                    time.sleep(1.0)
                    if time.time() - start > 60.0:
                        raise RuntimeError("No engine created. Failed to start the heartbeat process.")
                self.communicator.send_heartbeat(
                    self.servers, project_name, self.token, self.ssid, self.client_name, self.engine, interval
                )
        except FLCommunicationError:
            self.communicator.heartbeat_done = True

    def quit_remote(self, project_name, fl_ctx: FLContext):
        """Sending the last message to the server before leaving.

        Args:
            fl_ctx: FLContext

        Returns: N/A

        """
        return self.communicator.quit_remote(self.servers, project_name, self.token, self.ssid, fl_ctx)

    def heartbeat(self, interval):
        """Sends a heartbeat from the client to the server."""
        pool = None
        try:
            pool = ThreadPool(len(self.servers))
            return pool.map(partial(self.send_heartbeat, interval=interval), tuple(self.servers))
        finally:
            if pool:
                pool.terminate()

    def pull_task(self, fl_ctx: FLContext):
        """Fetch remote models and update the local client's session."""
        pool = None
        try:
            pool = ThreadPool(len(self.servers))
            self.remote_tasks = pool.map(partial(self.fetch_execute_task, fl_ctx=fl_ctx), tuple(self.servers))
            pull_success, task_name, shareable = _check_progress(self.remote_tasks)
            # TODO: if some of the servers failed
            return pull_success, task_name, shareable
        finally:
            if pool:
                pool.terminate()

    def push_results(self, shareable: Shareable, fl_ctx: FLContext):
        """Push the local model to multiple servers."""
        pool = None
        try:
            pool = ThreadPool(len(self.servers))
            return pool.map(partial(self.push_execute_result, shareable=shareable, fl_ctx=fl_ctx), tuple(self.servers))
        finally:
            if pool:
                pool.terminate()

    def register(self):
        """Push the local model to multiple servers."""
        pool = None
        try:
            pool = ThreadPool(len(self.servers))
            return pool.map(self.client_register, tuple(self.servers))
        finally:
            if pool:
                pool.terminate()

    def set_primary_sp(self, sp):
        pool = None
        try:
            pool = ThreadPool(len(self.servers))
            return pool.map(partial(self.set_sp, sp=sp), tuple(self.servers))
        finally:
            if pool:
                pool.terminate()

    def run_heartbeat(self, interval):
        """Periodically runs the heartbeat."""
        try:
            self.heartbeat(interval)
        except:
            self.logger.error("Failed to start run_heartbeat.")

    def start_heartbeat(self, interval=30):
        heartbeat_thread = threading.Thread(target=self.run_heartbeat, args=[interval])
        heartbeat_thread.start()

    def logout_client(self, fl_ctx: FLContext):
        """Logout the client from the server.

        Args:
            fl_ctx: FLContext

        Returns: N/A

        """
        pool = None
        try:
            pool = ThreadPool(len(self.servers))
            return pool.map(partial(self.quit_remote, fl_ctx=fl_ctx), tuple(self.servers))
        finally:
            if pool:
                pool.terminate()

    def set_client_engine(self, engine):
        self.engine = engine

    def set_client_runner(self, client_runner):
        self.client_runner = client_runner

    def stop_cell(self):
        """Stop the cell communication"""
        if self.communicator.cell:
            self.communicator.cell.stop()

    def close(self):
        """Quit the remote federated server, close the local session."""
        self.terminate()

        if self.engine:
            fl_ctx = self.engine.new_context()
        else:
            fl_ctx = FLContext()
        self.logout_client(fl_ctx)
        self.logger.info(f"Logout client: {self.client_name} from server.")

        return 0

    def terminate(self):
        """Terminating the local client session."""
        self.logger.info(f"Shutting down client run: {self.client_name}")
        if self.overseer_agent:
            self.overseer_agent.end()

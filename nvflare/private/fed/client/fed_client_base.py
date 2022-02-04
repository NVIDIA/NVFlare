# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

"""The client of the federated training process."""

import logging
import threading
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from typing import List, Optional

from nvflare.apis.filter import Filter
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import FLCommunicationError
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.private.defs import EngineConstant

from .client_status import ClientStatus
from .communicator import Communicator


class FederatedClientBase:
    """Federated client-side base implementation.

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
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.client_name = client_name
        self.token = None
        self.client_args = client_args
        self.servers = server_args

        self.communicator = Communicator(
            ssl_args=client_args,
            secure_train=secure_train,
            retry_timeout=retry_timeout,
            client_state_processors=client_state_processors,
            compression=compression,
        )

        self.secure_train = secure_train
        self.handlers = handlers

        self.heartbeat_done = False
        self.fl_ctx = FLContext()
        self.platform = None
        self.abort_signal = Signal()

        self.status = ClientStatus.NOT_STARTED

    def client_register(self, project_name):
        """Register the client to the FL server and get the FL token.

        Args:
            project_name: FL server project name

        Returns: N/A

        """
        if not self.token:
            try:
                self.token = self.communicator.client_registration(self.client_name, self.servers, project_name)
                if self.token is not None:
                    self.fl_ctx.set_prop(FLContextKey.CLIENT_NAME, self.client_name, private=False)
                    self.fl_ctx.set_prop(EngineConstant.FL_TOKEN, self.token, private=False)
                    self.logger.info(
                        "Successfully registered client:{} for {}. Got token:{}".format(
                            self.client_name, project_name, self.token
                        )
                    )
                    # fire_event(EventType.CLIENT_REGISTER, self.handlers, self.fl_ctx)

            except FLCommunicationError as e:
                self.communicator.heartbeat_done = True

    def fetch_execute_task(self, project_name, fl_ctx: FLContext):
        """Get registered with the remote server via channel, and fetch the server's model parameters.

        Args:
            project_name: FL study project name
            fl_ctx: FLContext

        Returns:  a CurrentTask message from server

        """
        try:
            self.logger.info("Starting to fetch execute task.")
            task = self.communicator.getTask(self.servers, project_name, self.token, fl_ctx)

            return task
        except FLCommunicationError as e:
            self.logger.info(e)
            # self.communicator.heartbeat_done = True

    def push_execute_result(self, project_name, shareable: Shareable, fl_ctx: FLContext):
        """Read local model and push to self.server[task_name] channel.

        This function makes and sends a Contribution Message.

        Args:
            project_name: FL study project name
            shareable: Shareable object
            fl_ctx: FLContext

        Returns: reply message

        """
        try:
            self.logger.info("Starting to push execute result.")
            execute_task_name = fl_ctx.get_prop(FLContextKey.TASK_NAME)
            message = self.communicator.submitUpdate(
                self.servers, project_name, self.token, fl_ctx, self.client_name, shareable, execute_task_name
            )

            return message
        except FLCommunicationError as e:
            self.logger.info(e)
            # self.communicator.heartbeat_done = True

    def send_aux_message(self, project_name, topic: str, shareable: Shareable, timeout: float, fl_ctx: FLContext):
        """Read local model and push to self.server[task_name] channel.

        This function makes and sends a Contribution Message.

        Args:
            project_name: FL study project name
            topic: aux topic name
            shareable: Shareable object
            timeout: communication timeout
            fl_ctx: FLContext

        Returns: reply message

        """
        try:
            self.logger.info("Starting to send aux messagee.")
            message = self.communicator.auxCommunicate(
                self.servers, project_name, self.token, fl_ctx, self.client_name, shareable, topic, timeout
            )

            return message
        except FLCommunicationError as e:
            self.logger.info(e)
            # self.communicator.heartbeat_done = True

    def send_heartbeat(self, project_name):
        try:
            if self.token:
                self.communicator.send_heartbeat(self.servers, project_name, self.token, self.client_name)
        except FLCommunicationError as e:
            self.communicator.heartbeat_done = True

    def heartbeat(self):
        """Sends a heartbeat from the client to the server."""
        pool = None
        try:
            pool = ThreadPool(len(self.servers))
            return pool.map(self.send_heartbeat, tuple(self.servers))
        finally:
            if pool:
                pool.terminate()

    def pull_task(self, fl_ctx: FLContext):
        """Fetch remote models and update the local client's session."""
        pool = None
        try:
            pool = ThreadPool(len(self.servers))
            self.remote_tasks = pool.map(partial(self.fetch_execute_task, fl_ctx=fl_ctx), tuple(self.servers))
            pull_success, task_name = self.check_progress(self.remote_tasks)
            # # Update app_ctx's current round info
            # if self.app_context and self.remote_models[0] is not None:
            #     self.app_context.global_round = self.remote_models[0].meta.current_round
            # TODO: if some of the servers failed
            # return self.model_manager.assign_current_model(self.remote_models)
            return pull_success, task_name, self.remote_tasks
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

    def aux_send(self, topic, shareable: Shareable, timeout: float, fl_ctx: FLContext):
        """Push the local model to multiple servers."""
        pool = None
        try:
            pool = ThreadPool(len(self.servers))
            messages = pool.map(
                partial(self.send_aux_message, topic=topic, shareable=shareable, timeout=timeout, fl_ctx=fl_ctx),
                tuple(self.servers),
            )
            if messages is not None and messages[0] is not None:
                # Only handle single server communication for now.
                return messages
            else:
                return None
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

    def run_heartbeat(self):
        """Periodically runs the heartbeat."""
        self.heartbeat()

    def start_heartbeat(self):
        heartbeat_thread = threading.Thread(target=self.run_heartbeat)
        heartbeat_thread.start()

    def quit_remote(self, task_name, fl_ctx: FLContext):
        """Sending the last message to the server before leaving.

        Args:
            task_name: task name
            fl_ctx: FLContext

        Returns: N/A

        """
        return self.communicator.quit_remote(self.servers, task_name, self.token, fl_ctx)

    def close(self):
        """Quit the remote federated server, close the local session."""
        self.logger.info("Shutting down client")

        return 0

    def check_progress(self, remote_tasks):
        if remote_tasks[0] is not None:
            self.server_meta = remote_tasks[0].meta
            return True, remote_tasks[0].task_name
        else:
            return False, None

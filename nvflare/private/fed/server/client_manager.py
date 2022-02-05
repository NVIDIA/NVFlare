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

import logging
import threading
import time
import uuid

import grpc

from nvflare.apis.client import Client


class ClientManager:
    def __init__(self, project_name=None, min_num_clients=2, max_num_clients=10):
        """Manages client adding and removing.

        Args:
            project_name: project name
            min_num_clients: minimum number of clients allowed.
            max_num_clients: maximum number of clients allowed.
        """
        self.project_name = project_name
        # TODO:: remove min num clients
        self.min_num_clients = min_num_clients
        self.max_num_clients = max_num_clients
        self.clients = dict()  # token => Client
        self.lock = threading.Lock()

        self.logger = logging.getLogger(self.__class__.__name__)

    def authenticate(self, request, context):
        client = self.login_client(request, context)
        if not client:
            return None

        client_ip = context.peer().split(":")[1]

        if len(self.clients) >= self.max_num_clients:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "Maximum number of clients reached")

        # new client will join the current round immediately
        with self.lock:
            self.clients.update({client.token: client})
            self.logger.info(
                "Client: New client {} joined. Sent token: {}.  Total clients: {}".format(
                    request.client_name + "@" + client_ip, client.token, len(self.clients)
                )
            )
        return client.token

    def remove_client(self, token):
        """Remove a registered client.

        Args:
            token: client token

        Returns:
            The removed Client object
        """
        with self.lock:
            client = self.clients.pop(token)
            self.logger.info(
                "Client Name:{} \tToken: {} left.  Total clients: {}".format(client.name, token, len(self.clients))
            )
            return client

    def login_client(self, client_login, context):
        if not self.is_valid_task(client_login.meta.project):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Requested task does not match the current server task")
        return self.authenticated_client(client_login, context)

    def validate_client(self, client_state, context, allow_new=False):
        """Validate the client state message.

        Args:
            client_state: A ClientState message received by server
            context: gRPC connection context
            allow_new: whether to allow new client. Note that its task should still match server's.

        Returns:
             client id if it's a valid client
        """
        token = client_state.token
        if not token:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Could not read client uid from the payload")
            client = None
        elif not self.is_valid_task(client_state.meta.project):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Requested task does not match the current server task")
            client = None
        elif not (allow_new or self.is_from_authorized_client(token)):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Unknown client identity")
            client = None
        else:
            client = self.clients.get(token)
        return client

    def authenticated_client(self, client_login, context) -> Client:
        """Use SSL certificate for authenticate the client.

        Args:
            client_login: client login request
            context: gRPC connection context

        Returns:
            Client object.
        """
        client = self.clients.get(client_login.token)
        if not client:
            cn_names = context.auth_context().get("x509_common_name")
            if cn_names:
                client_name = cn_names[0].decode("utf-8")
                if client_login.client_name:
                    if not client_login.client_name == client_name:
                        context.abort(
                            grpc.StatusCode.UNAUTHENTICATED, "client ID does not match the SSL certificate CN"
                        )
                        return None
            else:
                client_name = client_login.client_name

            for token, client in self.clients.items():
                if client.name == client_name:
                    context.abort(
                        grpc.StatusCode.FAILED_PRECONDITION,
                        "Client ID already registered as a client: {}".format(client_name),
                    )
                    return None

            client = Client(client_name, str(uuid.uuid4()))
        return client

    def is_from_authorized_client(self, token):
        """Check if a client is authorized.

        Args:
            token: client token

        Returns:
            True if it is a recognised client
        """
        return token in self.clients

    def is_valid_task(self, task):
        """Check whether the requested task matches the server's project_name.

        Returns:
            True if task name is the same as server's project name.
        """
        return task.name == self.project_name

    def heartbeat(self, token, client_name, context):
        """Update the heartbeat of the client.

        Args:
            token: client token
            client_name: client name
            context: grpc context

        Returns:
            If a new client needs to be created.
        """
        with self.lock:
            client = self.clients.get(token)
            if client:
                client.last_connect_time = time.time()
                # self.clients.update({token: time.time()})
                self.logger.debug(f"Receive heartbeat from Client:{token}")
                return False
            else:
                for _token, _client in self.clients.items():
                    if _client.name == client_name:
                        context.abort(
                            grpc.StatusCode.FAILED_PRECONDITION,
                            "Client ID already registered as a client: {}".format(client_name),
                        )
                        self.logger.info(
                            "Failed to re-activate dead client:{} with token: {}. Client already exist.".format(
                                client_name, _token
                            )
                        )
                        return False

                client = Client(client_name, token)
                client.last_connect_time = time.time()
                # self._set_instance_name(client)
                self.clients.update({token: client})
                self.logger.info("Re-activate dead client:{} with token: {}".format(client_name, token))

                return True

    def get_clients(self):
        """Get the list of registered clients.

        Returns:
            A dict of {client_token: client}
        """
        return self.clients

    def get_min_clients(self):
        return self.min_num_clients

    def get_max_clients(self):
        return self.max_num_clients

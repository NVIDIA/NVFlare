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

"""This is the FLAdmin Client to send the request message to the admin server."""

import threading
from multiprocessing.dummy import Pool as ThreadPool

import grpc

import nvflare.private.fed.protos.admin_pb2 as admin_msg
import nvflare.private.fed.protos.admin_pb2_grpc as admin_service
from nvflare.private.admin_defs import Message
from nvflare.private.fed.utils.messageproto import message_to_proto, proto_to_message

from .admin import Sender

lock = threading.Lock()


class AdminMessageSender(Sender):
    """AdminMessageSender to send the request message to the admin server."""

    def __init__(
        self,
        client_name,
        root_cert=None,
        ssl_cert=None,
        private_key=None,
        server_args=None,
        secure=False,
        is_multi_gpu=False,
        rank=0,
    ):
        """To init the AdminMessageSender.

        Args:
            client_name: client name
            root_cert: root certificate
            ssl_cert: SSL certificate
            private_key: private key
            server_args: server args
            secure: True/False
            is_multi_gpu: True/False
            rank: local process rank
        """
        self.client_name = client_name
        self.root_cert = root_cert
        self.ssl_cert = ssl_cert
        self.private_key = private_key
        self.secure = secure
        self.servers = server_args
        self.multi_gpu = is_multi_gpu
        self.rank = rank

        self.pool = ThreadPool(len(self.servers))

    def send_reply(self, message: Message):
        """Call to send the request message.

        Args:
            message: request message

        """
        if self.rank == 0:
            # self.send_client_reply(message)
            for taskname in tuple(self.servers):
                self._send_client_reply(message, taskname)

    def _send_client_reply(self, message, taskname):
        try:
            with self._set_up_channel(self.servers[taskname]) as channel:
                stub = admin_service.AdminCommunicatingStub(channel)

                reply = admin_msg.Reply()
                reply.client_name = self.client_name
                reply.message.CopyFrom(message_to_proto(message))
                # reply.message = message_to_proto(message)
                stub.SendReply(reply)
        except BaseException:
            pass

    def retrieve_requests(self) -> [Message]:
        """Send the message to retrieve pending requests from the Server.

        Returns: list of messages.

        """
        messages = []
        if self.rank == 0:
            items = self.pool.map(self._retrieve_client_requests, tuple(self.servers))
            for item in items:
                messages.extend(item)

        return messages

    def _retrieve_client_requests(self, taskname):
        try:
            message_list = []
            with self._set_up_channel(self.servers[taskname]) as channel:
                stub = admin_service.AdminCommunicatingStub(channel)

                client = admin_msg.Client()
                client.client_name = self.client_name
                messages = stub.Retrieve(client)
                for i in messages.message:
                    message_list.append(proto_to_message(i))
        except Exception as e:
            messages = None
        return message_list

    def send_result(self, message: Message):
        """Send the processor results to server.

        Args:
            message: message

        """
        if self.rank == 0:
            for taskname in tuple(self.servers):
                try:
                    with self._set_up_channel(self.servers[taskname]) as channel:
                        stub = admin_service.AdminCommunicatingStub(channel)

                        reply = admin_msg.Reply()
                        reply.client_name = self.client_name
                        reply.message.CopyFrom(message_to_proto(message))
                        stub.SendResult(reply)
                except BaseException:
                    pass

    def _set_up_channel(self, channel_dict):
        """Connect client to the server.

        Args:
            channel_dict: grpc channel parameters

        Returns: an initialised grpc channel

        """
        if self.secure:
            with open(self.root_cert, "rb") as f:
                trusted_certs = f.read()
            with open(self.private_key, "rb") as f:
                private_key = f.read()
            with open(self.ssl_cert, "rb") as f:
                certificate_chain = f.read()

            call_credentials = grpc.metadata_call_credentials(
                lambda context, callback: callback((("x-custom-token", self.client_name),), None)
            )
            credentials = grpc.ssl_channel_credentials(
                certificate_chain=certificate_chain, private_key=private_key, root_certificates=trusted_certs
            )

            composite_credentials = grpc.composite_channel_credentials(credentials, call_credentials)
            channel = grpc.secure_channel(**channel_dict, credentials=composite_credentials)
        else:
            channel = grpc.insecure_channel(**channel_dict)
        return channel

    def close(self):
        self.pool.close()

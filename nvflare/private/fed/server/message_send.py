# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.fuel.f3.cellnet.cell import FQCN
from nvflare.fuel.f3.cellnet.cell import Message as CellMessage
from nvflare.fuel.f3.cellnet.cell import TargetMessage
from nvflare.private.admin_defs import Message
from nvflare.private.defs import CellChannel, new_cell_message


class ClientReply(object):
    def __init__(self, client_token: str, req: Message, reply: Message):
        """Client reply.

        Args:
            client_token (str): client token
            req (Message): request
            reply (Message): reply
        """
        self.client_token = client_token
        self.request = req
        self.reply = reply


def send_requests(
    cell, command: str, requests: dict, clients, job_id=None, timeout_secs=2.0, optional=False
) -> [ClientReply]:
    """Send requests to clients.

    NOTE::

        This method is to be used by a Command Handler to send requests to Clients.
        Hence, it is run in the Command Handler's handling thread.
        This is a blocking call - returned only after all responses are received or timeout.

    Args:
        cell: the source cell
        command: the command to be sent
        clients: the clients the command will be sent to
        requests: A dict of requests: {client token: request or list of requests}
        job_id: id of the job that the command is applied to
        timeout_secs: how long to wait for reply before timeout
        optional: whether the message is optional

    Returns:
        A list of ClientReply
    """

    if not isinstance(requests, dict):
        raise TypeError("requests must be a dict but got {}".format(type(requests)))

    if len(requests) == 0:
        return []

    target_msgs = {}
    name_to_token = {}
    name_to_req = {}
    for token, req in requests.items():
        client = clients.get(token)
        if not client:
            continue

        if job_id:
            fqcn = FQCN.join([client.name, job_id])
            channel = CellChannel.CLIENT_COMMAND
            optional = True
        else:
            fqcn = client.name
            channel = CellChannel.CLIENT_MAIN
        target_msgs[client.name] = TargetMessage(
            target=fqcn, channel=channel, topic=command, message=new_cell_message({}, req)
        )

        name_to_token[client.name] = token
        name_to_req[client.name] = req

    if not target_msgs:
        return []

    if timeout_secs <= 0.0:
        # this is fire-and-forget!
        cell.fire_multi_requests_and_forget(target_msgs, optional=optional)
        return []
    else:
        result = []
        replies = cell.broadcast_multi_requests(target_msgs, timeout_secs, optional=optional)
        for name, reply in replies.items():
            assert isinstance(reply, CellMessage)
            result.append(ClientReply(client_token=name_to_token[name], req=name_to_req[name], reply=reply.payload))
        return result

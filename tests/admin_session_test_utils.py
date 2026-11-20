# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import uuid
from types import SimpleNamespace

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessagePropKey
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.admin_name_utils import new_admin_client_name
from nvflare.private.defs import ADMIN_TRANSPORT_TOKEN_PREFIX, CellMessageHeaderKeys


def register_admin_transport(
    manager, *, user="admin@example.com", origin=None, fingerprint=b"admin-cert", cert_exp=9999999999, connection=None
):
    origin = origin or new_admin_client_name()
    token = ADMIN_TRANSPORT_TOKEN_PREFIX + str(uuid.uuid4())
    request = Message(
        headers={
            MessageHeaderKey.ORIGIN: origin,
            CellMessageHeaderKeys.TOKEN: token,
            CellMessageHeaderKeys.CLIENT_NAME: user,
        }
    )
    request.set_prop(MessagePropKey.ENDPOINT, SimpleNamespace(name=origin))
    if connection is None:
        request.set_prop(MessagePropKey.PEER_CERT_FINGERPRINT, fingerprint)
    else:
        request.set_prop(MessagePropKey.CONNECTION, connection)
    assert manager.register_transport(request, user, token, fingerprint, cert_exp)
    grant = manager.get_transport(request)
    assert grant is not None
    return request, grant

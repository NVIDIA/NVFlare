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
from nvflare.apis.fl_constant import CellMessageAuthHeaderKey
from nvflare.fuel.f3.message import Message


def add_authentication_headers(msg: Message, client_name: str, auth_token, token_signature):
    """Add authentication headers to the specified message.

    Args:
        msg: the message that the headers are added to
        client_name: name of the client
        auth_token: authentication token
        token_signature: token signature

    Returns:

    """
    if client_name:
        msg.set_header(CellMessageAuthHeaderKey.CLIENT_NAME, client_name)

    msg.set_header(CellMessageAuthHeaderKey.TOKEN, auth_token if auth_token else "NA")
    msg.set_header(CellMessageAuthHeaderKey.TOKEN_SIGNATURE, token_signature if token_signature else "NA")

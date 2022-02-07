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

import uuid


class MsgHeader(object):

    REF_MSG_ID = "_refMsgId"
    RETURN_CODE = "_rtnCode"


class ReturnCode(object):

    OK = "_ok"
    ERROR = "_error"


class Message(object):
    def __init__(self, topic: str, body):
        """To init a Message.

        Args:
            topic: message topic
            body: message body.
        """
        self.id = str(uuid.uuid4())
        self.topic = topic
        self.body = body
        self.headers = {}

    def set_header(self, key, value):
        self.headers[key] = value

    def set_headers(self, headers: dict):
        if not headers:
            return
        if not isinstance(headers, dict):
            raise TypeError("headers must be dict but got {}".format(type(headers)))
        if len(headers) > 0:
            self.headers.update(headers)

    def get_header(self, key, default=None):
        return self.headers.get(key, default)

    def get_ref_id(self, default=None):
        return self.get_header(MsgHeader.REF_MSG_ID, default)

    def set_ref_id(self, msg_id):
        self.set_header(MsgHeader.REF_MSG_ID, msg_id)


def error_reply(err: str) -> Message:
    msg = Message(topic="reply", body=err)
    msg.set_header(MsgHeader.RETURN_CODE, ReturnCode.ERROR)
    return msg


def ok_reply(data=None) -> Message:
    if data is None:
        data = "ok"

    msg = Message(topic="reply", body=data)
    msg.set_header(MsgHeader.RETURN_CODE, ReturnCode.OK)
    return msg

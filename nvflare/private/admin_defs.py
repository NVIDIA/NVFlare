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

from nvflare.apis.shareable import Shareable, ReturnCode


class AdminMessage(Shareable):
    def __init__(self, topic: str, body):
        """To init a Message.

        Args:
            topic: message topic
            body: message body.
        """
        Shareable.__init__(self)
        self['body'] = body
        self.set_header('topic', topic)

    def set_headers(self, headers: dict):
        if not headers:
            return
        if not isinstance(headers, dict):
            raise TypeError("headers must be dict but got {}".format(type(headers)))
        if len(headers) > 0:
            for k, v in headers:
                self.set_header(k, v)

    @property
    def body(self):
        return self.get('body')

    @property
    def topic(self):
        return self.get_header('topic')


def error_reply(err: str) -> AdminMessage:
    msg = AdminMessage(topic="reply", body=err)
    msg.set_return_code(ReturnCode.ERROR)
    return msg


def ok_reply(topic=None, body=None) -> AdminMessage:
    if body is None:
        body = "ok"

    if topic is None:
        topic = "reply"

    msg = AdminMessage(topic=topic, body=body)
    msg.set_return_code(ReturnCode.OK)
    return msg

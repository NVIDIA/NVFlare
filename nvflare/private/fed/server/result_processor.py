# Copyright (c) 2021, NVIDIA CORPORATION.
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

from nvflare.private.admin_defs import Message


class ResultProcessor(object):
    """
    The RequestProcessor is responsible for processing a request.
    """

    def get_topics(self) -> [str]:
        """
        Get topics that this processor will handle
        :return: list of topics
        """
        pass

    def process(self, client_name, req: Message):
        """
        Called to process the specified request
        :param req:
        :param app_ctx:
        :return: a reply message
        """
        pass


class ValidateResultProcessor(ResultProcessor):
    def get_topics(self) -> [str]:
        return ["validate"]

    def process(self, client_name, message: Message):

        print(
            "Got the client: {} processor result. topic: {} \tbody: {}".format(
                client_name, message.topic, str(message.body)
            )
        )

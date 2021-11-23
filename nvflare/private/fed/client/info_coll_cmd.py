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

import json

from nvflare.private.admin_defs import Message
from nvflare.private.defs import InfoCollectorTopic
from nvflare.private.fed.client.admin import RequestProcessor
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec


class ClientInfoProcessor(RequestProcessor):
    def get_topics(self) -> [str]:
        return [
            InfoCollectorTopic.SHOW_STATS,
            InfoCollectorTopic.SHOW_ERRORS,
            InfoCollectorTopic.RESET_ERRORS,
        ]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        assert isinstance(engine, ClientEngineInternalSpec)
        # run_info = engine.get_current_run_info()
        # run_num = -1
        # if not run_info or run_info.run_number < 0:
        #     result = {
        #         'error': 'app not running',
        #     }
        # else:
        #     collector = engine.get_widget(WidgetID.INFO_COLLECTOR)
        #     if not collector:
        #         result = {
        #             'error': 'no info collector'
        #         }
        #     else:
        #         assert isinstance(collector, InfoCollector)
        #         run_num = run_info.run_number
        #
        #         if req.topic == InfoCollectorTopic.SHOW_STATS:
        #             result = collector.get_run_stats()
        #         elif req.topic == InfoCollectorTopic.SHOW_ERRORS:
        #             result = collector.get_errors()
        #         elif req.topic == InfoCollectorTopic.RESET_ERRORS:
        #             collector.reset_errors()
        #             result = {
        #                 'status': 'OK'
        #             }
        #         else:
        #             result = {
        #                 'error': 'invalid topic {}'.format(req.topic)
        #             }

        if req.topic == InfoCollectorTopic.SHOW_STATS:
            result = engine.get_current_run_info()
        elif req.topic == InfoCollectorTopic.SHOW_ERRORS:
            result = engine.get_errors()
        elif req.topic == InfoCollectorTopic.RESET_ERRORS:
            engine.reset_errors()
            result = {"status": "OK"}
        else:
            result = {"error": "invalid topic {}".format(req.topic)}

        if not isinstance(result, dict):
            result = {}

        # # add current run number
        # if run_num >= 0:
        #     result['run_num'] = run_num

        result = json.dumps(result)
        return Message(topic="reply_" + req.topic, body=result)

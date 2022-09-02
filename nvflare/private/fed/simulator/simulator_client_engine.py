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

from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.private.fed.client.client_engine import ClientEngine


class SimulatorClientEngine(ClientEngine):
    def __init__(self, client, client_name, sender, args, rank, workers=5):
        super().__init__(client, client_name, sender, args, rank, workers)

    def send_aux_command(self, shareable: Shareable, job_id):
        run_manager = self.client.run_manager
        if run_manager:
            with run_manager.new_context() as fl_ctx:
                topic = shareable.get_header(ReservedHeaderKey.TOPIC)
                return run_manager.dispatch(topic=topic, request=shareable, fl_ctx=fl_ctx)

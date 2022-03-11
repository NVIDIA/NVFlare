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

from gunicorn.workers.sync import SyncWorker


class ClientAuthWorker(SyncWorker):
    def handle_request(self, listener, req, client, addr):
        cert = client.getpeercert()
        subject = client.getpeercert().get("subject")
        commonName = next(value for ((key, value),) in subject if key == "commonName")
        headers = dict(req.headers)
        headers["X-USER"] = commonName
        req.headers = list(headers.items())

        super(ClientAuthWorker, self).handle_request(listener, req, client, addr)

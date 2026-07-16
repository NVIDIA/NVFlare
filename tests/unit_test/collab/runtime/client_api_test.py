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

import threading
from concurrent.futures import ThreadPoolExecutor

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.collab.runtime.client_api import CollabClientAPI


def test_concurrent_clients_use_their_own_api_context():
    barrier = threading.Barrier(2)
    observed_site_names = {}

    def make_api(site_name):
        api = CollabClientAPI()
        api._sys_info["site_name"] = site_name

        def train():
            flare.init()
            barrier.wait(timeout=5)
            observed_site_names[site_name] = flare.get_site_name()
            input_model = flare.receive()
            flare.send(FLModel(params={"site_name": site_name, "input": input_model.params}))

        api.set_training_func(train)
        return api

    apis = [make_api("site-1"), make_api("site-2")]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(api.execute, FLModel(params={"round": 1})) for api in apis]
        results = [future.result(timeout=10) for future in futures]

    assert observed_site_names == {"site-1": "site-1", "site-2": "site-2"}
    assert [result.params["site_name"] for result in results] == ["site-1", "site-2"]

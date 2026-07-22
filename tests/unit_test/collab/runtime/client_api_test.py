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
from unittest.mock import MagicMock, patch

import pytest

import nvflare.client as flare
from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE, AnalyticsDataType
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.collab.api.app import ClientApp
from nvflare.collab.runtime.client_api import CollabClientAPI
from nvflare.collab.runtime.flare.executor import CollabExecutor


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


def test_log_requires_initialized_site_runtime():
    api = CollabClientAPI()

    with pytest.raises(RuntimeError, match="site runtime is initialized"):
        api.log("loss", 0.5, AnalyticsDataType.SCALAR)


def test_log_forwards_analytics_to_runtime_handler():
    api = CollabClientAPI()
    handler = MagicMock(return_value="logged")
    api.set_log_handler(handler)

    result = api.log("loss", 0.5, AnalyticsDataType.SCALAR, global_step=3)

    assert result == "logged"
    handler.assert_called_once_with("loss", 0.5, AnalyticsDataType.SCALAR, global_step=3)


def test_training_function_flare_log_forwards_to_runtime_handler():
    api = CollabClientAPI()
    handler = MagicMock()
    api.set_log_handler(handler)

    def train():
        flare.init()
        model = flare.receive()
        flare.log("loss", 0.5, AnalyticsDataType.SCALAR, global_step=3)
        flare.send(model)

    api.set_training_func(train)
    result = api.execute(FLModel(params={"round": 1}))

    assert result.params == {"round": 1}
    handler.assert_called_once_with("loss", 0.5, AnalyticsDataType.SCALAR, global_step=3)


def test_execute_resets_receive_before_send_guard_between_rounds():
    api = CollabClientAPI()
    round_number = 0

    def train():
        nonlocal round_number
        round_number += 1
        flare.init()
        if round_number == 1:
            model = flare.receive()
            flare.send(model, clear_cache=False)
        else:
            flare.send(FLModel(params={"round": round_number}), clear_cache=False)

    api.set_training_func(train)
    first_result = api.execute(FLModel(params={"round": 1}))

    assert first_result.params == {"round": 1}
    with pytest.raises(RuntimeError, match='"receive" must be called before "send"'):
        api.execute(FLModel(params={"round": 2}))


def test_executor_configures_collab_client_api_logging():
    api = CollabClientAPI()
    app = ClientApp(api)
    executor = CollabExecutor(client_obj_id="client")
    handler = MagicMock()
    executor._log_analytic_data = handler

    try:
        executor._configure_client_api_logging(app)
        api.log("loss", 0.5, AnalyticsDataType.SCALAR)
    finally:
        executor.thread_executor.shutdown()

    handler.assert_called_once_with("loss", 0.5, AnalyticsDataType.SCALAR)


def test_executor_emits_client_api_analytics_event():
    executor = CollabExecutor(client_obj_id="client")
    executor._engine = MagicMock()
    fl_ctx = executor._engine.new_context.return_value.__enter__.return_value
    dxo = MagicMock()

    try:
        with patch("nvflare.collab.runtime.flare.executor.create_analytic_dxo", return_value=dxo) as create_dxo:
            with patch("nvflare.collab.runtime.flare.executor.send_analytic_dxo") as send_dxo:
                executor._log_analytic_data("loss", 0.5, AnalyticsDataType.SCALAR, global_step=3)
    finally:
        executor.thread_executor.shutdown()

    create_dxo.assert_called_once_with(tag="loss", value=0.5, data_type=AnalyticsDataType.SCALAR, global_step=3)
    send_dxo.assert_called_once_with(
        executor,
        dxo=dxo,
        fl_ctx=fl_ctx,
        event_type=ANALYTIC_EVENT_TYPE,
        fire_fed_event=False,
    )

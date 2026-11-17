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

import json
import os
from types import SimpleNamespace

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.default_component_policy import DEFAULT_CLASS_ALLOW_LIST
from nvflare.app_opt.catalog.classic import (
    REPLY_KEY,
    TASK_NAME,
    TASK_TIMEOUT_S,
    CatalogController,
    CatalogExecutor,
    CatalogRecipe,
)

from .catalog_test_utils import CONTRACT, FakeFLContext, write_site_config


def test_executor_replies_with_the_contracts_in_the_sites_local_folder(tmp_path):
    write_site_config(tmp_path / "local", {"omop": {"contract": "omop.odcs.yaml"}})

    reply = CatalogExecutor().execute(TASK_NAME, Shareable(), FakeFLContext(tmp_path), Signal())

    assert reply.get_return_code() == ReturnCode.OK
    assert reply[REPLY_KEY] == {"contracts": {"omop": CONTRACT}, "unavailable": []}


def test_controller_records_every_site_of_the_job(tmp_path, monkeypatch):
    controller = CatalogController()
    controller._engine = SimpleNamespace(get_clients=lambda: [SimpleNamespace(name=n) for n in ("a", "b", "c")])
    ok = make_reply(ReturnCode.OK)
    ok[REPLY_KEY] = {"contracts": {"omop": CONTRACT}, "unavailable": []}
    sent = {}

    def broadcast_and_wait(task, fl_ctx, min_responses, abort_signal):
        sent.update(name=task.name, timeout=task.timeout, min_responses=min_responses)
        for name, result in (("a", ok), ("b", make_reply(ReturnCode.EXECUTION_EXCEPTION))):
            task.result_received_cb(SimpleNamespace(client=SimpleNamespace(name=name), result=result), fl_ctx)

    monkeypatch.setattr(controller, "broadcast_and_wait", broadcast_and_wait)

    controller.control_flow(Signal(), FakeFLContext(tmp_path))

    assert sent == {"name": TASK_NAME, "timeout": TASK_TIMEOUT_S, "min_responses": 3}
    catalog_dir = tmp_path / "job-1" / "catalog"
    sites = json.loads((catalog_dir / "sites.json").read_text())
    assert {site: entry["status"] for site, entry in sites.items()} == {"a": "OK", "b": "FAILED", "c": "FAILED"}
    assert sorted(os.listdir(catalog_dir)) == ["a.omop.odcs.yaml", "sites.json"]


def test_job_components_run_without_byoc():
    recipe = CatalogRecipe()

    assert recipe is not None
    for component in (CatalogController, CatalogExecutor):
        assert f"{component.__module__}.{component.__name__}" in DEFAULT_CLASS_ALLOW_LIST

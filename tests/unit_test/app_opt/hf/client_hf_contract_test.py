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

from ._helpers import ClientAPIMock, import_hf_module, patch_client_api_aliases


def test_client_hf_reexports_standard_client_api_surface(monkeypatch):
    hf_client = import_hf_module(monkeypatch, "nvflare.client.hf")

    import nvflare.client as flare

    assert hf_client.AnalyticsDataType is flare.AnalyticsDataType
    assert hf_client.FLModel is flare.FLModel
    assert hf_client.IPCAgent is flare.IPCAgent
    assert hf_client.ParamsType is flare.ParamsType
    assert callable(hf_client.patch)

    for name in (
        "get_config",
        "get_job_id",
        "get_site_name",
        "get_task_name",
        "init",
        "is_evaluate",
        "is_submit_model",
        "is_train",
        "log",
        "receive",
        "send",
        "shutdown",
        "system_info",
    ):
        assert hasattr(hf_client, name), f"nvflare.client.hf must export {name}"
        assert getattr(hf_client, name) is getattr(flare, name)

    assert hf_client.is_running is not flare.is_running


def test_client_hf_is_running_delegates_before_a_trainer_is_patched(monkeypatch):
    client_api_mock = ClientAPIMock(running=True)
    patch_client_api_aliases(monkeypatch, client_api_mock)
    hf_client = import_hf_module(monkeypatch, "nvflare.client.hf")
    patch_client_api_aliases(monkeypatch, client_api_mock, hf_client)

    assert hf_client.is_running()
    assert client_api_mock.events == ["is_running"]

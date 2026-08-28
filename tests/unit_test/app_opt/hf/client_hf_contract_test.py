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

import pytest

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

    assert hf_client.init is not flare.init
    assert hf_client.is_running is not flare.is_running


def _clear_rank_environment(monkeypatch, hf_client):
    monkeypatch.delenv("RANK", raising=False)
    for name in hf_client._MULTIRANK_SIZE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_client_hf_init_keeps_single_process_rankless(monkeypatch):
    hf_client = import_hf_module(monkeypatch, "nvflare.client.hf")
    calls = []
    _clear_rank_environment(monkeypatch, hf_client)
    monkeypatch.setattr(hf_client, "_get_initialized_distributed_rank", lambda: None)
    monkeypatch.setattr(
        hf_client,
        "_client_api_init",
        lambda *, rank, config_file: calls.append((rank, config_file)) or "context",
    )

    assert hf_client.init(config_file="config.json") == "context"
    assert calls == [(None, "config.json")]


def test_client_hf_init_uses_initialized_distributed_rank(monkeypatch):
    hf_client = import_hf_module(monkeypatch, "nvflare.client.hf")
    calls = []
    _clear_rank_environment(monkeypatch, hf_client)
    monkeypatch.setattr(hf_client, "_get_initialized_distributed_rank", lambda: 1)
    monkeypatch.setattr(
        hf_client,
        "_client_api_init",
        lambda *, rank, config_file: calls.append((rank, config_file)) or "context",
    )

    assert hf_client.init() == "context"
    assert calls == [(1, None)]


def test_client_hf_init_uses_global_rank_before_delayed_process_group_init(monkeypatch):
    hf_client = import_hf_module(monkeypatch, "nvflare.client.hf")
    calls = []
    _clear_rank_environment(monkeypatch, hf_client)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setattr(hf_client, "_get_initialized_distributed_rank", lambda: None)
    monkeypatch.setattr(
        hf_client,
        "_client_api_init",
        lambda *, rank, config_file: calls.append((rank, config_file)) or "context",
    )

    assert hf_client.init() == "context"
    assert calls == [("1", None)]


@pytest.mark.parametrize(("raw_rank", "expected_rank"), (("+0", "0"), ("00", "0"), ("01", "1")))
def test_client_hf_init_normalizes_environment_global_rank(monkeypatch, raw_rank, expected_rank):
    hf_client = import_hf_module(monkeypatch, "nvflare.client.hf")
    calls = []
    _clear_rank_environment(monkeypatch, hf_client)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", raw_rank)
    monkeypatch.setattr(hf_client, "_get_initialized_distributed_rank", lambda: None)
    monkeypatch.setattr(
        hf_client,
        "_client_api_init",
        lambda *, rank, config_file: calls.append((rank, config_file)) or "context",
    )

    assert hf_client.init() == "context"
    assert calls == [(expected_rank, None)]


@pytest.mark.parametrize("size_marker", ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS"))
def test_client_hf_init_rejects_unresolved_multirank_before_client_context(monkeypatch, size_marker):
    hf_client = import_hf_module(monkeypatch, "nvflare.client.hf")
    calls = []
    _clear_rank_environment(monkeypatch, hf_client)
    monkeypatch.setenv(size_marker, "2")
    monkeypatch.setattr(hf_client, "_get_initialized_distributed_rank", lambda: None)
    monkeypatch.setattr(
        hf_client,
        "_client_api_init",
        lambda *, rank, config_file: calls.append((rank, config_file)),
    )

    with pytest.raises(RuntimeError, match="global RANK is unavailable"):
        hf_client.init()

    assert calls == []


def test_client_hf_init_rejects_negative_environment_global_rank_before_client_context(monkeypatch):
    hf_client = import_hf_module(monkeypatch, "nvflare.client.hf")
    calls = []
    _clear_rank_environment(monkeypatch, hf_client)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "-1")
    monkeypatch.setattr(hf_client, "_get_initialized_distributed_rank", lambda: None)
    monkeypatch.setattr(
        hf_client,
        "_client_api_init",
        lambda *, rank, config_file: calls.append((rank, config_file)),
    )

    with pytest.raises(RuntimeError, match="valid non-negative global RANK"):
        hf_client.init()

    assert calls == []


def test_client_hf_init_accepts_explicit_rank_in_multirank_environment(monkeypatch):
    hf_client = import_hf_module(monkeypatch, "nvflare.client.hf")
    calls = []
    _clear_rank_environment(monkeypatch, hf_client)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setattr(hf_client, "_get_initialized_distributed_rank", lambda: None)
    monkeypatch.setattr(
        hf_client,
        "_client_api_init",
        lambda *, rank, config_file: calls.append((rank, config_file)) or "context",
    )

    assert hf_client.init(rank=1) == "context"
    assert calls == [(1, None)]


def test_client_hf_is_running_delegates_before_a_trainer_is_patched(monkeypatch):
    client_api_mock = ClientAPIMock(running=True)
    patch_client_api_aliases(monkeypatch, client_api_mock)
    hf_client = import_hf_module(monkeypatch, "nvflare.client.hf")
    patch_client_api_aliases(monkeypatch, client_api_mock, hf_client)

    assert hf_client.is_running()
    assert client_api_mock.events == ["is_running"]

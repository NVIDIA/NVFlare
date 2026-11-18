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
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

wandb = pytest.importorskip("wandb")

from nvflare.apis.fl_constant import ProcessType
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.tracking.wandb import wandb_receiver as subject


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_BASE_URL", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    ctx = MagicMock()
    ctx.get_workspace.return_value = Workspace(str(tmp_path))
    ctx.get_peer_context.return_value = None
    ctx.get_prop.return_value = None
    ctx.get_process_type.return_value = ProcessType.SERVER_JOB
    ctx.get_engine.return_value.get_clients.return_value = [
        SimpleNamespace(name="site-1"),
        SimpleNamespace(name="site-2"),
    ]
    ctx.get_job_id.return_value = "job-1"
    return ctx


def args(**updates):
    return {"name": "test-run", "project": "test", "group": "test", "job_type": "train", **updates}


def approve(context, value, default=False):
    workspace = context.get_workspace()
    from pathlib import Path

    path = Path(workspace.get_site_config_dir()) / ("resources.json.default" if default else "resources.json")
    path.write_text(json.dumps({"wandb_base_urls": value}))


def capture_workers(monkeypatch):
    queues, processes = [], []

    def make_queue():
        queue = MagicMock()
        queues.append(queue)
        return queue

    def make_process(**kwargs):
        process = MagicMock()
        processes.append(process)
        return process

    monkeypatch.setattr(subject, "Queue", make_queue)
    monkeypatch.setattr(subject, "Process", make_process)
    monkeypatch.setattr(subject, "time", SimpleNamespace(time=lambda: 123, sleep=lambda _: None))
    return queues, processes


def test_default_backend_is_explicitly_pinned_to_sdk_default(context):
    config = subject.WandBReceiver(args())._get_wandb_args(context)
    assert config["settings"]["base_url"] == wandb.Settings().base_url
    assert config["settings"]["insecure_disable_ssl"] is False


@pytest.mark.parametrize("mode", ["online", "run", "offline", "disabled"])
@pytest.mark.parametrize("placement", [ProcessType.SERVER_JOB, ProcessType.CLIENT_JOB])
@pytest.mark.parametrize("key_present", [False, True])
def test_unapproved_backend_stops_before_login_queue_or_process(context, monkeypatch, mode, placement, key_present):
    context.get_process_type.return_value = placement
    if key_present:
        monkeypatch.setenv("WANDB_API_KEY", "a" * 40)
    login = MagicMock()
    queue = MagicMock()
    process = MagicMock()
    monkeypatch.setattr(subject.wandb, "login", login)
    monkeypatch.setattr(subject, "Queue", queue)
    monkeypatch.setattr(subject, "Process", process)
    receiver = subject.WandBReceiver(args(settings={"base_url": "http://127.0.0.1:19999"}), mode=mode)
    with pytest.raises(ValueError, match="approval in site wandb_base_urls"):
        receiver.initialize(context)
    login.assert_not_called()
    queue.assert_not_called()
    process.assert_not_called()


def test_operator_environment_supplies_default_and_cannot_be_replaced_by_job(context, monkeypatch):
    monkeypatch.setenv("WANDB_BASE_URL", "https://trusted.example/")
    config = subject.WandBReceiver(args())._get_wandb_args(context)
    assert config["settings"]["base_url"] == "https://trusted.example"
    with pytest.raises(ValueError, match="approval"):
        subject.WandBReceiver(args(settings={"base_url": "https://other.example"}))._get_wandb_args(context)


@pytest.mark.parametrize("default", [False, True])
def test_operator_approval_accepts_exact_backend_and_sdk_settings_objects(context, default):
    approve(context, ["https://tracking.example/"], default=default)
    settings = wandb.Settings(base_url="https://tracking.example")
    receiver = subject.WandBReceiver(args(settings=settings))
    result = receiver._get_wandb_args(context)
    assert result["settings"].base_url == "https://tracking.example"
    assert result["settings"] is not settings


@pytest.mark.parametrize(
    "url", ["https://tracking.example.evil", "http://tracking.example", "https://tracking.example/other"]
)
def test_approval_has_no_prefix_scheme_or_path_wildcards(context, url):
    approve(context, ["https://tracking.example"])
    with pytest.raises(ValueError, match="approval"):
        subject.WandBReceiver(args(settings={"base_url": url}))._get_wandb_args(context)


@pytest.mark.parametrize(
    "url",
    [
        "https://u:p@tracking.example",
        "file:///tmp/store",
        "https://tracking.example?q=1",
        "https://tracking.example/#x",
        "https://tracking.example\n",
        123,
    ],
)
def test_invalid_urls_fail_closed(context, url):
    with pytest.raises(ValueError):
        subject.WandBReceiver(args(settings={"base_url": url}))._get_wandb_args(context)


@pytest.mark.parametrize("value", ["https://tracking.example", [None], [False], [""], ["file:///tmp/store"], {}])
def test_malformed_operator_approval_fails_closed(context, value):
    approve(context, value)
    with pytest.raises(ValueError):
        subject.WandBReceiver(args())._get_wandb_args(context)


def test_job_resources_cannot_grant_backend_approval(context, tmp_path, monkeypatch):
    job = tmp_path / "job-1" / "app_server" / "config"
    job.mkdir(parents=True)
    (job / "resources.json").write_text(json.dumps({"wandb_base_urls": ["https://job.example"]}))
    (tmp_path / "local" / "resources.json").write_text("{}")
    (tmp_path / "local" / "resources.json.default").write_text(json.dumps({"wandb_base_urls": ["https://job.example"]}))
    monkeypatch.chdir(job)
    with pytest.raises(ValueError, match="approval"):
        subject.WandBReceiver(args(settings={"base_url": "https://job.example"}))._get_wandb_args(context)


@pytest.mark.parametrize(
    "setting,value",
    [
        ("http_proxy", "http://proxy.example"),
        ("https_proxy", "http://proxy.example"),
        ("x_proxies", {"http": "http://proxy.example"}),
        ("_proxies", {"https": "http://proxy.example"}),
        ("insecure_disable_ssl", True),
        ("insecure_disable_ssl", "false"),
    ],
)
def test_job_cannot_bypass_backend_policy_through_transport_overrides(context, setting, value):
    with pytest.raises(ValueError, match="proxies or disable TLS"):
        subject.WandBReceiver(args(settings={setting: value}))._get_wandb_args(context)


def test_approved_online_login_and_worker_init_use_same_destination(context, monkeypatch):
    approve(context, ["https://approved.example"])
    login = MagicMock()
    monkeypatch.setattr(subject.wandb, "login", login)
    queues, processes = capture_workers(monkeypatch)
    original = args(settings={"base_url": "https://approved.example/"}, config={"learning_rate": 0.1})
    snapshot = deepcopy(original)
    receiver = subject.WandBReceiver(original, mode="online")
    receiver.initialize(context)
    login.assert_called_once_with(host="https://approved.example", timeout=1, verify=True)
    assert len(queues) == len(processes) == 2
    first, second = [q.put.call_args.args[0].task_data for q in queues]
    assert first["settings"]["base_url"] == second["settings"]["base_url"] == "https://approved.example"
    assert first["name"] == "site-1-job-1-test-run"
    assert second["name"] == "site-2-job-1-test-run"
    assert first["config"]["client"] == "site-1"
    assert second["config"]["client"] == "site-2"
    assert original == snapshot
    for process in processes:
        process.start.assert_called_once()
    original["settings"]["base_url"] = "https://changed.example"
    assert first["settings"]["base_url"] == "https://approved.example"


def test_offline_mode_preserves_no_login_behavior(context, monkeypatch):
    login = MagicMock()
    monkeypatch.setattr(subject.wandb, "login", login)
    queues, _ = capture_workers(monkeypatch)
    subject.WandBReceiver(args(), mode="offline").initialize(context)
    login.assert_not_called()
    assert queues[0].put.call_args.args[0].task_data["mode"] == "offline"


def test_settings_object_without_explicit_backend_inherits_site_default(context, monkeypatch):
    monkeypatch.setenv("WANDB_BASE_URL", "https://trusted.example")
    settings = wandb.Settings(init_timeout=3)
    result = subject.WandBReceiver(args(settings=settings))._get_wandb_args(context)
    assert result["settings"].base_url == "https://trusted.example"
    assert "base_url" in result["settings"].model_fields_set
    assert result["settings"].init_timeout == 3

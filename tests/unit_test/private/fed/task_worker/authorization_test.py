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
from types import SimpleNamespace

import pytest

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.common.excepts import ComponentNotAuthorized
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity, WorkerBootstrap
from nvflare.private.fed.task_worker.worker import _authorize_bootstrap


@pytest.fixture
def authorize(tmp_path):
    ConfigService.reset()
    resources = tmp_path / "resources.json"
    resources.write_text(json.dumps({"class_allow_list": ["allowed."]}))
    workspace = SimpleNamespace(get_resources_file_path=lambda: str(resources))
    ctx = FLContext()
    ctx.set_prop(FLContextKey.JOB_META, {"byoc": False})

    def check(executor, components=()):
        bootstrap = WorkerBootstrap(
            TaskAttemptIdentity("job", "site", "task", "train", "attempt"),
            str(tmp_path),
            str(tmp_path),
            executor,
            components=components,
        )
        _authorize_bootstrap(bootstrap, workspace, ctx)

    yield check, ctx, resources
    ConfigService.reset()


def test_worker_blocks_executor_before_import(authorize):
    check, _ctx, _resources = authorize
    with pytest.raises(ComponentNotAuthorized, match="forbidden.Executor.*executor.*allow_list"):
        check({"path": "forbidden.Executor"})


@pytest.mark.parametrize(
    "args",
    [
        {"helper": {"path": "forbidden.Helper"}},
        {"helpers": [{"class_path": "forbidden.Helper"}]},
        {"path": "ordinary-argument", "helper": {"path": "forbidden.Helper"}},
    ],
)
def test_worker_rejects_nested_construction(authorize, args):
    check, _ctx, _resources = authorize
    with pytest.raises(ValueError, match="nested components are not supported"):
        check({"path": "allowed.Executor", "args": args})


def test_worker_preserves_plain_constructor_data(authorize):
    check, _ctx, _resources = authorize
    check(
        {
            "path": "allowed.Executor",
            "args": {
                "path": "ordinary-argument",
                "name": "ordinary-name",
                "options": {"config_type": "dict", "path": "ordinary.data", "numbers": [1, 2]},
            },
        }
    )


def test_only_decomposer_registration_is_repeated_in_worker(authorize):
    check, _ctx, resources = authorize
    resources.write_text(json.dumps({"class_allow_list": ["allowed.", "nvflare."]}))
    check(
        {"path": "allowed.Executor"},
        components=(
            {
                "id": "codecs",
                "path": "nvflare.app_common.widgets.decomposer_reg.DecomposerRegister",
                "args": {"classes": []},
            },
        ),
    )
    with pytest.raises(ValueError, match="only decomposer registration"):
        check({"path": "allowed.Executor"}, components=({"id": "helper", "path": "allowed.Helper"},))


def test_explicit_component_list_entries_cannot_bypass_site_policy(authorize):
    check, _ctx, resources = authorize
    resources.write_text(json.dumps({"class_allow_list": ["allowed.", "nvflare."]}))
    with pytest.raises(ComponentNotAuthorized, match="components.#2"):
        check(
            {"path": "allowed.Executor"},
            components=(
                {
                    "id": "first",
                    "path": "nvflare.app_common.widgets.decomposer_reg.DecomposerRegister",
                    "args": {"classes": []},
                },
                {"id": "helper", "config_type": "dict", "path": "forbidden.Helper"},
            ),
        )


def test_byoc_uses_authoritative_context_metadata(authorize):
    check, ctx, _resources = authorize
    ctx.set_prop(FLContextKey.JOB_META, {"byoc": True})
    check({"path": "custom.Executor"})


def test_missing_site_policy_uses_current_default_allowlist(authorize):
    check, _ctx, resources = authorize
    resources.write_text("{}")
    check({"path": "nvflare.app_common.np.np_trainer.NPTrainer"})
    with pytest.raises(ComponentNotAuthorized, match="allow_list"):
        check({"path": "subprocess.Popen"})


def test_warn_policy_is_preserved(authorize):
    check, _ctx, resources = authorize
    resources.write_text(json.dumps({"class_allow_list": [], "class_list_enforcement_mode": "warn"}))
    check({"path": "unlisted.Executor"})


@pytest.mark.parametrize("wrapped", [False, True])
def test_statistics_generator_is_authorized_before_import(authorize, wrapped):
    check, _ctx, resources = authorize
    resources.write_text(json.dumps({"class_allow_list": ["allowed.", "nvflare."]}))
    executor = {
        "path": "nvflare.app_common.executors.statistics.statistics_executor.StatisticsExecutor",
        "args": {"generator_id": "generator"},
    }
    if wrapped:
        executor["args"] = {"class_attributes": executor["args"]}
    check(executor, components=({"id": "generator", "path": "allowed.Statistics"},))
    with pytest.raises(ComponentNotAuthorized, match="forbidden.Statistics.*components.#1"):
        check(executor, components=({"id": "generator", "path": "forbidden.Statistics"},))
    with pytest.raises(ValueError, match="generator_id"):
        check(executor)


def test_wrapped_constructor_arguments_cannot_hide_nested_components(authorize):
    check, _ctx, _resources = authorize
    with pytest.raises(ValueError, match="nested components are not supported"):
        check(
            {
                "path": "allowed.Executor",
                "args": {
                    "class_attributes": {
                        "path": "ordinary-value",
                        "name": "ordinary-name",
                        "helper": {"path": "forbidden.Helper"},
                    }
                },
            }
        )

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

from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.executors.client_api import script_utils
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from nvflare.client.api_spec import CLIENT_API_KEY
from nvflare.client.config import ConfigKey


def test_shared_metadata_preserves_the_declared_exchange_contract():
    context = ClientAPIBackendContext(executor=Mock(), train_task_name="fit", train_with_evaluation=True)
    metadata = script_utils.prepare_task_metadata(context, "endpoint", "job", "fit")
    assert metadata[FLMetaKey.SITE_NAME] == "endpoint"
    assert metadata[FLMetaKey.JOB_ID] == "job"
    assert metadata[ConfigKey.TASK_NAME] == "fit"
    assert metadata[ConfigKey.TASK_EXCHANGE] == {
        ConfigKey.TRAIN_WITH_EVAL: True,
        ConfigKey.EXCHANGE_FORMAT: context.params_exchange_format,
        ConfigKey.SERVER_EXPECTED_FORMAT: context.server_expected_format,
        ConfigKey.TRANSFER_TYPE: context.params_transfer_type,
        ConfigKey.TRAIN_TASK_NAME: "fit",
        ConfigKey.EVAL_TASK_NAME: context.evaluate_task_name,
        ConfigKey.SUBMIT_MODEL_TASK_NAME: context.submit_model_task_name,
    }


@pytest.fixture
def script_binding(monkeypatch):
    context = ClientAPIBackendContext(executor=Mock(), task_script_path="train.py", task_script_args=["--flag"])
    api, runner, logger = Mock(), Mock(), Mock()
    decomposers, runner_factory = Mock(), Mock(return_value=runner)
    monkeypatch.setattr(script_utils, "register_framework_decomposers", decomposers)
    monkeypatch.setattr(script_utils, "TaskScriptRunner", runner_factory)
    return context, api, runner, logger, decomposers, runner_factory


@pytest.mark.parametrize("gc_rounds", [0, 3])
def test_shared_setup_does_not_start_the_script_or_bind_the_bus(script_binding, gc_rounds):
    context, api, runner, logger, decomposers, runner_factory = script_binding
    context = ClientAPIBackendContext(
        executor=context.executor, task_script_path="train.py", task_script_args=["--flag"], memory_gc_rounds=gc_rounds
    )
    api_factory, metadata = Mock(return_value=api), {"metadata": True}
    assert script_utils.create_script_binding(context, metadata, "custom", api_factory, logger) == (api, runner)
    api_factory.assert_called_once_with(metadata)
    api.init.assert_called_once_with()
    decomposers.assert_called_once_with(context.params_exchange_format, context.server_expected_format, logger)
    runner_factory.assert_called_once_with(custom_dir="custom", script_path="train.py", script_args=["--flag"])
    runner.run.assert_not_called()
    if gc_rounds:
        api.configure_memory_management.assert_called_once_with(gc_rounds=gc_rounds, cuda_empty_cache=False)
    else:
        api.configure_memory_management.assert_not_called()


@pytest.mark.parametrize("failure_point", ["init", "memory", "runner"])
def test_shared_setup_unwinds_partial_api_initialization(script_binding, failure_point):
    context, api, _runner, logger, _decomposers, runner_factory = script_binding
    context = ClientAPIBackendContext(executor=context.executor, task_script_path="train.py", memory_gc_rounds=1)
    failing_operation = {"init": api.init, "memory": api.configure_memory_management, "runner": runner_factory}
    failing_operation[failure_point].side_effect = RuntimeError("setup failed")
    api.close.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(RuntimeError, match="setup failed"):
        script_utils.create_script_binding(context, {}, "custom", Mock(return_value=api), logger)
    api.close.assert_called_once()
    assert "cleanup failed" in logger.error.call_args.args[0]


@pytest.mark.parametrize("failure_point", ["close", "lookup", "clear"])
@pytest.mark.parametrize("best_effort", [False, True])
def test_shared_cleanup_preserves_backend_failure_policy(failure_point, best_effort):
    api, bus, on_error = Mock(), Mock(), Mock()
    bus.get_data.return_value = api
    operations = {"close": api.close, "lookup": bus.get_data, "clear": bus.put_data}
    operations[failure_point].side_effect = RuntimeError("cleanup failed")
    if best_effort:
        script_utils.close_script_api(api, bus, on_error)
        assert "cleanup failed" in on_error.call_args.args[0]
    else:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            script_utils.close_script_api(api, bus)
    api.close.assert_called_once()
    bus.get_data.assert_called_once_with(CLIENT_API_KEY)
    if failure_point != "lookup":
        bus.put_data.assert_called_once_with(CLIENT_API_KEY, None)


def test_shared_cleanup_does_not_clear_a_different_api_owner():
    api, bus = Mock(), Mock()
    bus.get_data.return_value = object()
    script_utils.close_script_api(api, bus)
    api.close.assert_called_once()
    bus.put_data.assert_not_called()
    script_utils.close_script_api(None, bus)
    bus.put_data.assert_not_called()

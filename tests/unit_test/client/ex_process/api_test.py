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

import copy
from unittest.mock import patch

import pytest

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.params_converter import ParamsConverter
from nvflare.client.config import ClientConfig, ConfigKey, ExchangeFormat
from nvflare.client.converter_utils import create_default_params_converters
from nvflare.client.ex_process.api import _downgrade_rotating_handlers


class _DummyConverter(ParamsConverter):
    def convert(self, params, fl_ctx):
        return params


def _fake_optional_import(module: str, name: str):
    class _LocalDummyConverter(_DummyConverter):
        pass

    _LocalDummyConverter.__name__ = name
    return _LocalDummyConverter, True


def test_create_params_converters_for_pytorch_format():
    config = ClientConfig(
        {
            ConfigKey.TASK_EXCHANGE: {
                ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.PYTORCH,
                ConfigKey.SERVER_EXPECTED_FORMAT: ExchangeFormat.NUMPY,
                ConfigKey.TRAIN_TASK_NAME: "train",
                ConfigKey.EVAL_TASK_NAME: "validate",
                ConfigKey.SUBMIT_MODEL_TASK_NAME: "submit_model",
            }
        }
    )

    with patch("nvflare.client.converter_utils.optional_import", side_effect=_fake_optional_import):
        from_converter, to_converter = create_default_params_converters(
            server_expected_format=config.get_server_expected_format(),
            params_exchange_format=config.get_exchange_format(),
            train_task_name=config.get_train_task(),
            eval_task_name=config.get_eval_task(),
            submit_model_task_name=config.get_submit_model_task(),
        )

    assert from_converter is not None
    assert to_converter is not None
    assert from_converter.__class__.__name__ == "NumpyToPTParamsConverter"
    assert to_converter.__class__.__name__ == "PTToNumpyParamsConverter"
    assert from_converter.supported_tasks == ["train", "validate"]
    assert to_converter.supported_tasks == ["train", "submit_model"]


def test_create_params_converters_skips_when_server_expected_format_is_not_numpy():
    config = ClientConfig(
        {
            ConfigKey.TASK_EXCHANGE: {
                ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.PYTORCH,
                ConfigKey.SERVER_EXPECTED_FORMAT: ExchangeFormat.RAW,
                ConfigKey.TRAIN_TASK_NAME: "train",
                ConfigKey.EVAL_TASK_NAME: "validate",
                ConfigKey.SUBMIT_MODEL_TASK_NAME: "submit_model",
            }
        }
    )

    from_converter, to_converter = create_default_params_converters(
        server_expected_format=config.get_server_expected_format(),
        params_exchange_format=config.get_exchange_format(),
        train_task_name=config.get_train_task(),
        eval_task_name=config.get_eval_task(),
        submit_model_task_name=config.get_submit_model_task(),
    )
    assert from_converter is None
    assert to_converter is None


# ---------------------------------------------------------------------------
# Fix 3: ClientConfig.get_submit_result_timeout() and ExProcessClientAPI wiring
# ---------------------------------------------------------------------------


def test_get_submit_result_timeout_default():
    """When SUBMIT_RESULT_TIMEOUT is absent, default must be 300.0."""
    config = ClientConfig({ConfigKey.TASK_EXCHANGE: {}})
    assert config.get_submit_result_timeout() == 300.0


def test_get_submit_result_timeout_explicit():
    """Explicit value must be returned as a float."""
    config = ClientConfig({ConfigKey.TASK_EXCHANGE: {ConfigKey.SUBMIT_RESULT_TIMEOUT: 600.0}})
    assert config.get_submit_result_timeout() == 600.0


def test_get_submit_result_timeout_int_coerced_to_float():
    """Integer value in config must be coerced to float."""
    config = ClientConfig({ConfigKey.TASK_EXCHANGE: {ConfigKey.SUBMIT_RESULT_TIMEOUT: 120}})
    result = config.get_submit_result_timeout()
    assert isinstance(result, float)
    assert result == 120.0


def test_get_submit_result_timeout_absent_section():
    """Missing TASK_EXCHANGE section must still return the default 300.0."""
    config = ClientConfig({})
    assert config.get_submit_result_timeout() == 300.0


def test_get_streaming_idle_timeout_default_and_explicit():
    config = ClientConfig({})
    assert config.get_streaming_idle_timeout() == 600.0

    config = ClientConfig({ConfigKey.TASK_EXCHANGE: {ConfigKey.STREAMING_IDLE_TIMEOUT: 1200}})
    assert config.get_streaming_idle_timeout() == 1200.0

    config = ClientConfig({ConfigKey.TASK_EXCHANGE: {ConfigKey.STREAMING_IDLE_TIMEOUT: None}})
    assert config.get_streaming_idle_timeout() is None


def test_ex_process_api_passes_submit_result_timeout_to_agent(monkeypatch):
    """ExProcessClientAPI.init() must pass task-exchange timeouts from config to FlareAgentWithFLModel."""
    from unittest.mock import MagicMock

    from nvflare.client.config import ConfigKey
    from nvflare.client.ex_process.api import ExProcessClientAPI

    fake_config = ClientConfig(
        {
            ConfigKey.TASK_EXCHANGE: {
                ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.NUMPY,
                ConfigKey.SERVER_EXPECTED_FORMAT: ExchangeFormat.NUMPY,
                ConfigKey.TRAIN_TASK_NAME: "train",
                ConfigKey.EVAL_TASK_NAME: "validate",
                ConfigKey.SUBMIT_MODEL_TASK_NAME: "submit_model",
                ConfigKey.HEARTBEAT_TIMEOUT: 60,
                ConfigKey.SUBMIT_RESULT_TIMEOUT: 999.0,
                ConfigKey.STREAMING_IDLE_TIMEOUT: 1200.0,
            }
        }
    )

    mock_pipe = MagicMock()
    captured_kwargs = {}

    class _CapturingAgent:
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(
        "nvflare.client.ex_process.api._create_client_config",
        lambda config: fake_config,
    )
    monkeypatch.setattr(
        "nvflare.client.ex_process.api._create_pipe_using_config",
        lambda client_config, section: (mock_pipe, "task"),
    )
    monkeypatch.setattr(
        "nvflare.client.ex_process.api.FlareAgentWithFLModel",
        _CapturingAgent,
    )
    monkeypatch.setattr(
        "nvflare.client.ex_process.api.create_default_params_converters",
        lambda **kwargs: (None, None),
    )
    monkeypatch.setattr(
        "nvflare.client.ex_process.api.ModelRegistry",
        lambda *args, **kwargs: MagicMock(),
    )

    api = ExProcessClientAPI(config_file="fake_config.json")
    api._configure_subprocess_logging = lambda client_config: None
    api.init(rank="0")

    assert "submit_result_timeout" in captured_kwargs, "submit_result_timeout was not passed to FlareAgentWithFLModel"
    assert captured_kwargs["submit_result_timeout"] == 999.0
    assert captured_kwargs["streaming_idle_timeout"] == 1200.0


def test_ex_process_receive_timeout_does_not_arm_send_under_congestion():
    from unittest.mock import MagicMock

    from nvflare.client.config import ClientConfig
    from nvflare.client.ex_process.api import ExProcessClientAPI
    from nvflare.client.model_registry import ModelRegistry

    fake_agent = MagicMock()
    fake_agent.get_task.return_value = None
    registry = ModelRegistry(ClientConfig({}), rank="0", flare_agent=fake_agent)
    api = ExProcessClientAPI(config_file="fake_config.json")
    api.model_registry = registry

    assert api.receive(timeout=12.5) is None
    fake_agent.get_task.assert_called_once_with(12.5)
    assert api.receive_called is False
    with pytest.raises(RuntimeError, match='"receive" needs to be called'):
        api.send(FLModel(params={"x": 1}))


def test_ex_process_receive_timeout_then_later_task_allows_send():
    from unittest.mock import MagicMock

    from nvflare.client.config import ClientConfig
    from nvflare.client.ex_process.api import ExProcessClientAPI
    from nvflare.client.flare_agent import Task
    from nvflare.client.model_registry import ModelRegistry

    received_model = FLModel(params={"x": 1})
    fake_agent = MagicMock()
    fake_agent.get_task.side_effect = [None, Task(task_name="train", task_id="task-1", data=received_model)]
    registry = ModelRegistry(ClientConfig({}), rank="0", flare_agent=fake_agent)
    api = ExProcessClientAPI(config_file="fake_config.json")
    api.model_registry = registry

    assert api.receive(timeout=0.1) is None
    assert api.receive_called is False

    assert api.receive(timeout=0.1) is received_model
    assert api.receive_called is True
    api.send(FLModel(params={"x": 2}), clear_cache=False)
    fake_agent.submit_result.assert_called_once()


# ── _downgrade_rotating_handlers tests ────────────────────────────────────────

_ROTATING_CONFIG = {
    "handlers": {
        "logFileHandler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "baseFormatter",
            "filename": "log.txt",
            "mode": "a",
            "maxBytes": 20971520,
            "backupCount": 10,
        },
        "timedHandler": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "WARNING",
            "filename": "log_timed.txt",
            "when": "midnight",
            "interval": 1,
            "backupCount": 7,
            "utc": True,
        },
        "consoleHandler": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
    }
}


class TestDowngradeRotatingHandlers:
    def test_rotating_file_handler_replaced_with_file_handler(self):
        cfg = copy.deepcopy(_ROTATING_CONFIG)
        _downgrade_rotating_handlers(cfg)
        assert cfg["handlers"]["logFileHandler"]["class"] == "logging.FileHandler"

    def test_timed_rotating_file_handler_replaced_with_file_handler(self):
        cfg = copy.deepcopy(_ROTATING_CONFIG)
        _downgrade_rotating_handlers(cfg)
        assert cfg["handlers"]["timedHandler"]["class"] == "logging.FileHandler"

    def test_rotation_keys_removed(self):
        cfg = copy.deepcopy(_ROTATING_CONFIG)
        _downgrade_rotating_handlers(cfg)
        h = cfg["handlers"]["logFileHandler"]
        for key in ("maxBytes", "backupCount"):
            assert key not in h, f"rotation key '{key}' should be removed"

    def test_timed_rotation_keys_removed(self):
        cfg = copy.deepcopy(_ROTATING_CONFIG)
        _downgrade_rotating_handlers(cfg)
        h = cfg["handlers"]["timedHandler"]
        for key in ("when", "interval", "backupCount", "utc"):
            assert key not in h, f"rotation key '{key}' should be removed"

    def test_non_rotation_keys_preserved(self):
        cfg = copy.deepcopy(_ROTATING_CONFIG)
        _downgrade_rotating_handlers(cfg)
        h = cfg["handlers"]["logFileHandler"]
        assert h["level"] == "DEBUG"
        assert h["formatter"] == "baseFormatter"
        assert h["filename"] == "log.txt"
        assert h["mode"] == "a"

    def test_stream_handler_untouched(self):
        cfg = copy.deepcopy(_ROTATING_CONFIG)
        original = copy.deepcopy(cfg["handlers"]["consoleHandler"])
        _downgrade_rotating_handlers(cfg)
        assert cfg["handlers"]["consoleHandler"] == original

    def test_empty_handlers_section(self):
        cfg = {"handlers": {}}
        _downgrade_rotating_handlers(cfg)  # must not raise

    def test_no_handlers_key(self):
        cfg = {}
        _downgrade_rotating_handlers(cfg)  # must not raise

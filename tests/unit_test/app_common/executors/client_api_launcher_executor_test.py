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
import threading
import time as _real_time

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import make_reply
from nvflare.app_common.executors.client_api_launcher_executor import ClientAPILauncherExecutor
from nvflare.app_common.executors.launcher_executor import LauncherExecutor
from nvflare.app_common.executors.task_exchanger import TaskExchanger
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe


def _make_fake_cell_pipe():
    """Return an uninitialized CellPipe instance that passes isinstance checks.

    object.__new__ bypasses __init__ so no network infrastructure is needed.
    """
    return object.__new__(CellPipe)


class _FakeCoreCell:
    def __init__(self, initial_ctx=None):
        self.ctx = dict(initial_ctx or {})

    def update_fobs_context(self, props: dict):
        self.ctx.update(props)

    def get_fobs_context(self, props: dict = None):
        result = copy.copy(self.ctx)
        if props:
            result.update(props)
        return result


class _FakeCell:
    def __init__(self, initial_ctx=None):
        self.core_cell = _FakeCoreCell(initial_ctx)
        self.decode_pass_through_channels: set = set()


class _FakeEngine:
    def __init__(self, cell):
        self._cell = cell

    def get_cell(self):
        return self._cell


class _FakeFLContext:
    def __init__(self, cell, identity_name="test_site", job_id="test_job"):
        self._engine = _FakeEngine(cell)
        self._identity_name = identity_name
        self._job_id = job_id
        self._props = {}
        self._peer_ctx = None

    def get_engine(self):
        return self._engine

    def get_identity_name(self):
        return self._identity_name

    def get_job_id(self):
        return self._job_id

    def get_prop(self, key, default=None):
        return self._props.get(key, default)

    def set_prop(self, key, value, private=False, sticky=False):
        self._props[key] = value

    def get_peer_context(self):
        return self._peer_ctx


@pytest.fixture
def base_executor(monkeypatch):
    monkeypatch.setattr(ClientAPILauncherExecutor, "prepare_config_for_launch", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "initialize", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "finalize", lambda self, fl_ctx: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_warning", lambda self, fl_ctx, msg: None)
    executor = ClientAPILauncherExecutor(pipe_id="test_pipe")
    # Fix 7: PASS_THROUGH is only enabled when pipe is a CellPipe.
    # Set the pipe to a minimal CellPipe instance so PASS_THROUGH tests exercise
    # the correct code path.
    executor.pipe = _make_fake_cell_pipe()
    return executor


def test_launcher_converter_ids_warn_when_ignored(monkeypatch):
    warnings = []
    monkeypatch.setattr(LauncherExecutor, "log_warning", lambda self, fl_ctx, msg: warnings.append(msg))

    executor = LauncherExecutor(
        pipe_id="test_pipe", from_nvflare_converter_id="from_converter", to_nvflare_converter_id="to_converter"
    )
    fl_ctx = _FakeFLContext(_FakeCell())

    executor._init_converter(fl_ctx)

    assert len(warnings) == 1
    assert "ignored in LauncherExecutor" in warnings[0]
    assert executor._from_nvflare_converter is None
    assert executor._to_nvflare_converter is None


# ---------------------------------------------------------------------------
# Fix 3: submit_result_timeout wiring through executor
# ---------------------------------------------------------------------------


def test_submit_result_timeout_default_stored():
    """Default submit_result_timeout (300.0) must be stored on the executor."""
    executor = ClientAPILauncherExecutor(pipe_id="test_pipe")
    assert executor._submit_result_timeout == 300.0


def test_submit_result_timeout_custom_stored():
    """Custom submit_result_timeout must be stored exactly as given."""
    executor = ClientAPILauncherExecutor(pipe_id="test_pipe", submit_result_timeout=600.0)
    assert executor._submit_result_timeout == 600.0


def test_prepare_config_includes_submit_result_timeout(monkeypatch):
    """prepare_config_for_launch must include SUBMIT_RESULT_TIMEOUT in the task_exchange section."""
    from unittest.mock import MagicMock

    from nvflare.client.config import ConfigKey

    captured = {}

    def _fake_write(config_data, config_file_path):
        captured.update(config_data)

    monkeypatch.setattr("nvflare.app_common.executors.client_api_launcher_executor.write_config_to_file", _fake_write)
    monkeypatch.setattr(
        "nvflare.app_common.executors.client_api_launcher_executor.update_export_props",
        lambda config_data, fl_ctx: None,
    )

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe", submit_result_timeout=450.0)

    # Mock pipe.export() — needed by prepare_config_for_launch
    mock_pipe = MagicMock()
    mock_pipe.export.return_value = ("nvflare.some.PipeClass", {"arg1": "val1"})
    executor.pipe = mock_pipe
    executor.heartbeat_timeout = 60.0
    executor._train_with_evaluation = False
    executor._params_exchange_format = "numpy"
    executor._server_expected_format = "numpy"
    executor._params_transfer_type = "FULL"
    executor._train_task_name = "train"
    executor._evaluate_task_name = "validate"
    executor._submit_model_task_name = "submit_model"
    executor._memory_gc_rounds = 0
    executor._cuda_empty_cache = False
    executor._config_file_name = "client_api_config.json"

    fake_workspace = MagicMock()
    fake_workspace.get_app_config_dir.return_value = "/tmp/fake_dir"
    fake_engine = MagicMock()
    fake_engine.get_workspace.return_value = fake_workspace
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = fake_engine
    fl_ctx.get_job_id.return_value = "test_job"

    executor.get_pipe_channel_name = lambda: "task"

    executor.prepare_config_for_launch(fl_ctx)

    task_exchange = captured.get(ConfigKey.TASK_EXCHANGE, {})
    assert ConfigKey.SUBMIT_RESULT_TIMEOUT in task_exchange
    assert task_exchange[ConfigKey.SUBMIT_RESULT_TIMEOUT] == 450.0


def test_prepare_config_submit_result_timeout_default_value(monkeypatch):
    """When no custom submit_result_timeout is given, config must contain 300.0."""
    from unittest.mock import MagicMock

    from nvflare.client.config import ConfigKey

    captured = {}

    monkeypatch.setattr(
        "nvflare.app_common.executors.client_api_launcher_executor.write_config_to_file",
        lambda config_data, config_file_path: captured.update(config_data),
    )
    monkeypatch.setattr(
        "nvflare.app_common.executors.client_api_launcher_executor.update_export_props",
        lambda config_data, fl_ctx: None,
    )

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe")

    mock_pipe = MagicMock()
    mock_pipe.export.return_value = ("SomeClass", {})
    executor.pipe = mock_pipe
    executor.heartbeat_timeout = 60.0
    executor._train_with_evaluation = False
    executor._params_exchange_format = "numpy"
    executor._server_expected_format = "numpy"
    executor._params_transfer_type = "FULL"
    executor._train_task_name = "train"
    executor._evaluate_task_name = "validate"
    executor._submit_model_task_name = "submit_model"
    executor._memory_gc_rounds = 0
    executor._cuda_empty_cache = False
    executor._config_file_name = "client_api_config.json"

    fake_workspace = MagicMock()
    fake_workspace.get_app_config_dir.return_value = "/tmp/fake_dir"
    fake_engine = MagicMock()
    fake_engine.get_workspace.return_value = fake_workspace
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = fake_engine
    fl_ctx.get_job_id.return_value = "test_job"

    executor.get_pipe_channel_name = lambda: "task"

    executor.prepare_config_for_launch(fl_ctx)

    assert captured[ConfigKey.TASK_EXCHANGE][ConfigKey.SUBMIT_RESULT_TIMEOUT] == 300.0


# ---------------------------------------------------------------------------
# Fix 5: peer_read_timeout runtime override via add_client_config()
# ---------------------------------------------------------------------------

_GCV_MODULE = "nvflare.app_common.executors.client_api_launcher_executor.get_client_config_value"


def _make_gcv_stub(overrides: dict):
    """Return a get_client_config_value replacement that answers from *overrides*."""

    def _gcv(fl_ctx, key, default=None):
        return overrides.get(key, default)

    return _gcv


def test_peer_read_timeout_not_overridden_when_absent(monkeypatch):
    """When PEER_READ_TIMEOUT is absent from config, peer_read_timeout stays at its constructor default."""
    monkeypatch.setattr(ClientAPILauncherExecutor, "prepare_config_for_launch", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "initialize", lambda self, fl_ctx: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_warning", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(_GCV_MODULE, _make_gcv_stub({}))

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe", peer_read_timeout=300.0)
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    executor.initialize(fl_ctx)

    assert executor.peer_read_timeout == 300.0


def test_peer_read_timeout_overridden_from_config(monkeypatch):
    """When PEER_READ_TIMEOUT is in config, peer_read_timeout must be updated to the config value."""
    from nvflare.client.constants import PEER_READ_TIMEOUT

    monkeypatch.setattr(ClientAPILauncherExecutor, "prepare_config_for_launch", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "initialize", lambda self, fl_ctx: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_warning", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(_GCV_MODULE, _make_gcv_stub({PEER_READ_TIMEOUT: 1800}))

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe", peer_read_timeout=300.0)
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    executor.initialize(fl_ctx)

    assert executor.peer_read_timeout == 1800.0


def test_peer_read_timeout_invalid_raises(monkeypatch):
    """A non-positive PEER_READ_TIMEOUT must raise ValueError."""
    from nvflare.client.constants import PEER_READ_TIMEOUT

    monkeypatch.setattr(ClientAPILauncherExecutor, "prepare_config_for_launch", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "initialize", lambda self, fl_ctx: None)
    monkeypatch.setattr(_GCV_MODULE, _make_gcv_stub({PEER_READ_TIMEOUT: -1}))
    # log_error calls fire_event which checks isinstance(fl_ctx, FLContext); patch it out.
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_error", lambda self, fl_ctx, msg: None)

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe")
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    with pytest.raises(ValueError, match="PEER_READ_TIMEOUT must be positive"):
        executor.initialize(fl_ctx)


def test_peer_read_timeout_zero_raises(monkeypatch):
    """Zero PEER_READ_TIMEOUT must raise ValueError (not silently accepted)."""
    from nvflare.client.constants import PEER_READ_TIMEOUT

    monkeypatch.setattr(ClientAPILauncherExecutor, "prepare_config_for_launch", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "initialize", lambda self, fl_ctx: None)
    monkeypatch.setattr(_GCV_MODULE, _make_gcv_stub({PEER_READ_TIMEOUT: 0}))
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_error", lambda self, fl_ctx, msg: None)

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe")
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    with pytest.raises(ValueError, match="PEER_READ_TIMEOUT must be positive"):
        executor.initialize(fl_ctx)


def test_peer_read_timeout_and_external_pre_init_both_overridable(monkeypatch):
    """Both PEER_READ_TIMEOUT and EXTERNAL_PRE_INIT_TIMEOUT must be independently overridable."""
    from nvflare.client.constants import EXTERNAL_PRE_INIT_TIMEOUT, PEER_READ_TIMEOUT

    monkeypatch.setattr(ClientAPILauncherExecutor, "prepare_config_for_launch", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "initialize", lambda self, fl_ctx: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_warning", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(
        _GCV_MODULE,
        _make_gcv_stub({PEER_READ_TIMEOUT: 900, EXTERNAL_PRE_INIT_TIMEOUT: 120}),
    )

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe", peer_read_timeout=300.0, external_pre_init_timeout=60.0)
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    executor.initialize(fl_ctx)

    assert executor.peer_read_timeout == 900.0
    assert executor._external_pre_init_timeout == 120.0


# ---------------------------------------------------------------------------
# Fix 10: max_resends wiring
# ---------------------------------------------------------------------------


def test_max_resends_default_stored():
    """Default max_resends (3) must be stored on the executor."""
    executor = ClientAPILauncherExecutor(pipe_id="test_pipe")
    assert executor._max_resends == 3


def test_max_resends_custom_stored():
    """Custom max_resends must be stored exactly as given."""
    executor = ClientAPILauncherExecutor(pipe_id="test_pipe", max_resends=10)
    assert executor._max_resends == 10


def test_max_resends_none_stored():
    """max_resends=None (unlimited) must be stored as-is."""
    executor = ClientAPILauncherExecutor(pipe_id="test_pipe", max_resends=None)
    assert executor._max_resends is None


def test_prepare_config_includes_max_resends(monkeypatch):
    """prepare_config_for_launch must include MAX_RESENDS in the task_exchange section."""
    from unittest.mock import MagicMock

    from nvflare.client.config import ConfigKey

    captured = {}
    monkeypatch.setattr(
        "nvflare.app_common.executors.client_api_launcher_executor.write_config_to_file",
        lambda config_data, config_file_path: captured.update(config_data),
    )
    monkeypatch.setattr(
        "nvflare.app_common.executors.client_api_launcher_executor.update_export_props",
        lambda config_data, fl_ctx: None,
    )

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe", max_resends=5)
    mock_pipe = MagicMock()
    mock_pipe.export.return_value = ("nvflare.some.PipeClass", {})
    executor.pipe = mock_pipe
    executor.heartbeat_timeout = 60.0
    executor._train_with_evaluation = False
    executor._params_exchange_format = "numpy"
    executor._server_expected_format = "numpy"
    executor._params_transfer_type = "FULL"
    executor._train_task_name = "train"
    executor._evaluate_task_name = "validate"
    executor._submit_model_task_name = "submit_model"
    executor._memory_gc_rounds = 0
    executor._cuda_empty_cache = False
    executor._config_file_name = "client_api_config.json"
    executor.get_pipe_channel_name = lambda: "task"

    fake_workspace = MagicMock()
    fake_workspace.get_app_config_dir.return_value = "/tmp/fake_dir"
    fake_engine = MagicMock()
    fake_engine.get_workspace.return_value = fake_workspace
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = fake_engine
    fl_ctx.get_job_id.return_value = "test_job"

    executor.prepare_config_for_launch(fl_ctx)

    task_exchange = captured.get(ConfigKey.TASK_EXCHANGE, {})
    assert ConfigKey.MAX_RESENDS in task_exchange
    assert task_exchange[ConfigKey.MAX_RESENDS] == 5


def test_client_config_get_max_resends_default():
    """ClientConfig.get_max_resends() must return 3 when key is absent."""
    from nvflare.client.config import ClientConfig

    cfg = ClientConfig(config={})
    assert cfg.get_max_resends() == 3


def test_client_config_get_max_resends_from_config():
    """ClientConfig.get_max_resends() must return the configured value."""
    from nvflare.client.config import ClientConfig, ConfigKey

    cfg = ClientConfig(config={ConfigKey.TASK_EXCHANGE: {ConfigKey.MAX_RESENDS: 7}})
    assert cfg.get_max_resends() == 7


def test_client_config_get_max_resends_none():
    """ClientConfig.get_max_resends() must return None when explicitly set to None."""
    from nvflare.client.config import ClientConfig, ConfigKey

    cfg = ClientConfig(config={ConfigKey.TASK_EXCHANGE: {ConfigKey.MAX_RESENDS: None}})
    assert cfg.get_max_resends() is None


def test_ex_process_api_passes_max_resends_to_flare_agent(monkeypatch):
    """ExProcessClientAPI must pass max_resends from ClientConfig to FlareAgentWithFLModel."""
    from unittest.mock import MagicMock, patch

    from nvflare.client.config import ClientConfig, ConfigKey

    client_config = ClientConfig(
        config={
            ConfigKey.TASK_EXCHANGE: {
                ConfigKey.MAX_RESENDS: 5,
                ConfigKey.SUBMIT_RESULT_TIMEOUT: 300.0,
                ConfigKey.HEARTBEAT_TIMEOUT: 600.0,
            }
        }
    )

    captured_kwargs = {}

    def fake_agent_init(self, *args, **kwargs):
        captured_kwargs.update(kwargs)

    with patch("nvflare.client.flare_agent_with_fl_model.FlareAgentWithFLModel.__init__", fake_agent_init):
        from nvflare.client.flare_agent_with_fl_model import FlareAgentWithFLModel

        agent = object.__new__(FlareAgentWithFLModel)
        fake_agent_init(agent, pipe=MagicMock(), max_resends=client_config.get_max_resends())

    assert captured_kwargs.get("max_resends") == 5


# ---------------------------------------------------------------------------
# CJ-side RSS logging and memory GC (check_output_shareable / Fix 12)
# ---------------------------------------------------------------------------


def _make_shareable(round_num=1):
    """Return a minimal Shareable with CURRENT_ROUND set."""
    from nvflare.apis.shareable import Shareable
    from nvflare.app_common.app_constant import AppConstants

    s = Shareable()
    s.set_header(AppConstants.CURRENT_ROUND, round_num)
    return s


def test_check_output_shareable_returns_true_on_success(monkeypatch):
    """check_output_shareable must return True when the parent returns True."""
    from unittest.mock import MagicMock, patch

    executor = ClientAPILauncherExecutor(pipe_id="p")
    executor._memory_gc_rounds = 0

    monkeypatch.setattr(LauncherExecutor, "check_output_shareable", lambda self, tn, sh, ctx: True)
    with patch("nvflare.fuel.utils.mem_utils.log_rss"):
        result = executor.check_output_shareable("train", _make_shareable(), MagicMock())

    assert result is True


def test_check_output_shareable_returns_false_when_parent_fails(monkeypatch):
    """check_output_shareable must return False when the parent returns False
    without calling log_rss or cleanup."""
    from unittest.mock import MagicMock, patch

    executor = ClientAPILauncherExecutor(pipe_id="p")
    executor._memory_gc_rounds = 1

    monkeypatch.setattr(LauncherExecutor, "check_output_shareable", lambda self, tn, sh, ctx: False)
    with (
        patch("nvflare.fuel.utils.mem_utils.log_rss") as mock_log_rss,
        patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup,
    ):
        result = executor.check_output_shareable("train", _make_shareable(), MagicMock())

    assert result is False
    mock_log_rss.assert_not_called()
    mock_cleanup.assert_not_called()


def test_cj_cleanup_not_called_when_gc_rounds_zero(monkeypatch):
    """_maybe_cleanup_cj_memory must not call cleanup_memory when memory_gc_rounds=0."""
    from unittest.mock import MagicMock, patch

    executor = ClientAPILauncherExecutor(pipe_id="p")
    executor._memory_gc_rounds = 0

    monkeypatch.setattr(LauncherExecutor, "check_output_shareable", lambda self, tn, sh, ctx: True)
    with (
        patch("nvflare.fuel.utils.mem_utils.log_rss"),
        patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup,
    ):
        executor.check_output_shareable("train", _make_shareable(), MagicMock())

    mock_cleanup.assert_not_called()


def test_cj_cleanup_called_every_n_rounds(monkeypatch):
    """cleanup_memory must be called exactly once per memory_gc_rounds sends."""
    from unittest.mock import MagicMock, patch

    executor = ClientAPILauncherExecutor(pipe_id="p")
    executor._memory_gc_rounds = 3
    executor._cuda_empty_cache = False
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_warning", lambda self, fl_ctx, msg: None)

    monkeypatch.setattr(LauncherExecutor, "check_output_shareable", lambda self, tn, sh, ctx: True)
    with (
        patch("nvflare.fuel.utils.mem_utils.log_rss"),
        patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup,
    ):
        for i in range(6):
            executor.check_output_shareable("train", _make_shareable(i), MagicMock())

    assert mock_cleanup.call_count == 2  # rounds 3 and 6


def test_cj_cleanup_passes_cuda_empty_cache(monkeypatch):
    """cleanup_memory must be called with cuda_empty_cache=True when configured."""
    from unittest.mock import MagicMock, patch

    executor = ClientAPILauncherExecutor(pipe_id="p")
    executor._memory_gc_rounds = 1
    executor._cuda_empty_cache = True
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_warning", lambda self, fl_ctx, msg: None)

    monkeypatch.setattr(LauncherExecutor, "check_output_shareable", lambda self, tn, sh, ctx: True)
    with (
        patch("nvflare.fuel.utils.mem_utils.log_rss"),
        patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup,
    ):
        executor.check_output_shareable("train", _make_shareable(), MagicMock())

    mock_cleanup.assert_called_once_with(cuda_empty_cache=True)


# ---------------------------------------------------------------------------
# Timeout validation warnings (_validate_timeout_config)
# ---------------------------------------------------------------------------


def _make_validating_executor(monkeypatch, **executor_kwargs):
    """Return an executor whose initialize() runs _validate_timeout_config() and
    records all log_warning calls into a list for assertion."""
    warnings_emitted = []

    monkeypatch.setattr(ClientAPILauncherExecutor, "prepare_config_for_launch", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "initialize", lambda self, fl_ctx: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(
        ClientAPILauncherExecutor,
        "log_warning",
        lambda self, fl_ctx, msg: warnings_emitted.append(msg),
    )

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe", **executor_kwargs)
    executor.pipe = _make_fake_cell_pipe()
    return executor, warnings_emitted


def test_timeout_warning_min_dl_less_than_per_req(monkeypatch):
    """A warning must fire when min_download_timeout < streaming_per_request_timeout."""
    import nvflare.fuel.utils.app_config_utils as acu
    from nvflare.apis.fl_constant import ConfigVarName

    executor, warnings = _make_validating_executor(monkeypatch)
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    def _fake_get(name, default):
        if ConfigVarName.MIN_DOWNLOAD_TIMEOUT in name:
            return 60.0  # explicitly small → min_dl < per_req
        if ConfigVarName.STREAMING_PER_REQUEST_TIMEOUT in name:
            return 600.0
        return default

    monkeypatch.setattr(acu, "get_positive_float_var", _fake_get)
    executor.initialize(fl_ctx)

    assert any("min_download_timeout" in w and "streaming_per_request_timeout" in w for w in warnings), warnings


def test_no_timeout_warning_when_min_dl_ge_per_req(monkeypatch):
    """No warning when min_download_timeout >= streaming_per_request_timeout."""
    import nvflare.fuel.utils.app_config_utils as acu
    from nvflare.apis.fl_constant import ConfigVarName

    executor, warnings = _make_validating_executor(monkeypatch)
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    def _fake_get(name, default):
        if ConfigVarName.MIN_DOWNLOAD_TIMEOUT in name:
            return 700.0  # min_dl > per_req
        if ConfigVarName.STREAMING_PER_REQUEST_TIMEOUT in name:
            return 600.0
        return default

    monkeypatch.setattr(acu, "get_positive_float_var", _fake_get)
    executor.initialize(fl_ctx)

    assert not any("min_download_timeout" in w and "streaming_per_request_timeout" in w for w in warnings), warnings


def test_timeout_warning_submit_exceeds_min_dl(monkeypatch):
    """A warning must fire when submit_result_timeout > min_download_timeout."""
    import nvflare.fuel.utils.app_config_utils as acu
    from nvflare.apis.fl_constant import ConfigVarName

    executor, warnings = _make_validating_executor(monkeypatch, submit_result_timeout=400.0)
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    def _fake_get(name, default):
        if ConfigVarName.MIN_DOWNLOAD_TIMEOUT in name:
            return 300.0  # min_dl=300 < submit_result_timeout=400
        if ConfigVarName.STREAMING_PER_REQUEST_TIMEOUT in name:
            return 100.0  # keep per_req < min_dl so only the submit warning fires
        return default

    monkeypatch.setattr(acu, "get_positive_float_var", _fake_get)
    executor.initialize(fl_ctx)

    assert any("submit_result_timeout" in w for w in warnings), warnings


def test_timeout_warning_unbounded_max_resends(monkeypatch):
    """A warning must fire when max_resends is None (unbounded)."""
    import nvflare.fuel.utils.app_config_utils as acu
    from nvflare.apis.fl_constant import ConfigVarName

    executor, warnings = _make_validating_executor(monkeypatch, max_resends=None)
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    def _fake_get(name, default):
        # Return values that won't trigger other warnings
        if ConfigVarName.MIN_DOWNLOAD_TIMEOUT in name:
            return 700.0
        if ConfigVarName.STREAMING_PER_REQUEST_TIMEOUT in name:
            return 600.0
        return default

    monkeypatch.setattr(acu, "get_positive_float_var", _fake_get)
    executor.initialize(fl_ctx)

    assert any("max_resends" in w for w in warnings), warnings


def test_no_warning_when_all_timeouts_consistent(monkeypatch):
    """No warnings when all timeout constraints are satisfied and max_resends is bounded."""
    import nvflare.fuel.utils.app_config_utils as acu
    from nvflare.apis.fl_constant import ConfigVarName

    executor, warnings = _make_validating_executor(monkeypatch, submit_result_timeout=200.0, max_resends=3)
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    def _fake_get(name, default):
        if ConfigVarName.MIN_DOWNLOAD_TIMEOUT in name:
            return 700.0
        if ConfigVarName.STREAMING_PER_REQUEST_TIMEOUT in name:
            return 600.0
        return default

    monkeypatch.setattr(acu, "get_positive_float_var", _fake_get)
    executor.initialize(fl_ctx)

    assert warnings == [], warnings


# ---------------------------------------------------------------------------
# Fix 14: PASS_THROUGH direction contract — CJ pipe must NOT stamp the header
# ---------------------------------------------------------------------------


def test_cj_pipe_pass_through_on_send_false_after_initialize(monkeypatch):
    """CJ's CellPipe must NOT have pass_through_on_send=True after initialize().

    Regression guard for RuntimeError: Could not infer dtype of LazyDownloadRef.

    If CJ's pipe stamps PASS_THROUGH on outgoing task messages (forward path),
    the subprocess Adapter decodes with PASS_THROUGH=True, causing
    ViaDownloaderDecomposer to return LazyDownloadRef objects instead of real
    tensors.  User code (e.g. torch.as_tensor()) then crashes because it
    cannot infer the dtype of a LazyDownloadRef.

    Only the subprocess-side CellPipe (set in ExProcessClientAPI.init()) should
    have pass_through_on_send=True — for the reverse path so CJ creates
    LazyDownloadRef from subprocess results and forwards them to the server.
    """
    monkeypatch.setattr(ClientAPILauncherExecutor, "prepare_config_for_launch", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "initialize", lambda self, fl_ctx: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_warning", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(_GCV_MODULE, _make_gcv_stub({}))

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe")
    cell_pipe = _make_fake_cell_pipe()
    cell_pipe.pass_through_on_send = False  # default
    executor.pipe = cell_pipe

    fl_ctx = _FakeFLContext(_FakeCell())
    executor.initialize(fl_ctx)

    assert not cell_pipe.pass_through_on_send, (
        "CJ's CellPipe must NOT have pass_through_on_send=True after initialize().\n"
        "Stamping PASS_THROUGH on forward-path task messages causes the subprocess\n"
        "to decode model params as LazyDownloadRef, crashing torch.as_tensor()."
    )


def test_decomposer_prefix_default_is_numpy(monkeypatch):
    """Base class _decomposer_prefix() must return 'np_'."""
    executor = ClientAPILauncherExecutor(pipe_id="p")
    assert executor._decomposer_prefix() == "np_"


def test_pt_executor_decomposer_prefix_is_tensor():
    """PTClientAPILauncherExecutor._decomposer_prefix() must return 'tensor_'."""
    pytest.importorskip("torch")
    from nvflare.app_opt.pt.client_api_launcher_executor import PTClientAPILauncherExecutor

    executor = object.__new__(PTClientAPILauncherExecutor)
    assert executor._decomposer_prefix() == "tensor_"


# ---------------------------------------------------------------------------
# Deferred stop_task() coordination (large-model download fix)
# ---------------------------------------------------------------------------


class _FakeLauncher:
    """Minimal Launcher stub for deferred-stop tests."""

    def __init__(self, needs_deferred=True, statuses=None):
        self._needs_deferred = needs_deferred
        self._statuses = list(statuses or ["success"])  # LauncherRunStatus.COMPLETE_SUCCESS
        self._status_idx = 0
        self.stop_task_calls = []

    def needs_deferred_stop(self):
        return self._needs_deferred

    def check_run_status(self, task_name, fl_ctx):
        if self._status_idx < len(self._statuses):
            s = self._statuses[self._status_idx]
            self._status_idx += 1
            return s
        return "success"  # LauncherRunStatus.COMPLETE_SUCCESS

    def stop_task(self, task_name, fl_ctx, abort_signal):
        self.stop_task_calls.append(task_name)

    def launch_task(self, task_name, shareable, fl_ctx, abort_signal):
        return True

    def initialize(self, fl_ctx):
        pass

    def finalize(self, fl_ctx):
        pass


class _FakeAbortSignal:
    def __init__(self, triggered=False):
        self.triggered = triggered

    def trigger(self, msg=""):
        self.triggered = True


def _make_deferred_stop_executor(monkeypatch, stop_task_wait_timeout=5.0):
    """Return a LauncherExecutor wired for deferred-stop unit tests.

    - Logging is silenced.
    - _execute_launcher_method_in_thread_executor is replaced with a direct
      delegation to self.launcher so tests don't need a live ThreadPoolExecutor.
    - _wait_external_setup always returns True.
    - reset_peer_is_up_or_dead is a no-op.
    """
    executor = LauncherExecutor(pipe_id="test_pipe")
    executor._stop_task_wait_timeout = stop_task_wait_timeout
    executor._job_end = False

    for meth in ("log_info", "log_warning", "log_debug", "log_error"):
        monkeypatch.setattr(LauncherExecutor, meth, lambda self, fl_ctx, msg: None)

    def _mock_execute(self_inner, method_name, **kwargs):
        fn = getattr(self_inner.launcher, method_name, None)
        if fn is None:
            return None
        if method_name == "check_run_status":
            return fn(kwargs.get("task_name"), kwargs.get("fl_ctx"))
        elif method_name == "stop_task":
            return fn(kwargs.get("task_name"), kwargs.get("fl_ctx"), kwargs.get("abort_signal"))
        elif method_name == "launch_task":
            return fn(
                kwargs.get("task_name"), kwargs.get("shareable"), kwargs.get("fl_ctx"), kwargs.get("abort_signal")
            )
        return None

    monkeypatch.setattr(LauncherExecutor, "_execute_launcher_method_in_thread_executor", _mock_execute)
    monkeypatch.setattr(LauncherExecutor, "_wait_external_setup", lambda self, tn, fl_ctx, abort: True)
    monkeypatch.setattr(LauncherExecutor, "reset_peer_is_up_or_dead", lambda self: None)
    return executor


def test_deferred_stop_event_starts_set():
    """`_deferred_stop_event` must start set (no deferred stop in progress)."""
    executor = LauncherExecutor(pipe_id="test_pipe")
    assert executor._deferred_stop_event.is_set()


def test_deferred_stop_task_name_default_empty():
    """`_deferred_stop_task_name` must be initialized to empty string."""
    executor = LauncherExecutor(pipe_id="test_pipe")
    assert executor._deferred_stop_task_name == ""


def test_stop_task_wait_timeout_default_zero():
    """`_stop_task_wait_timeout` must default to 0.0 on base LauncherExecutor."""
    executor = LauncherExecutor(pipe_id="test_pipe")
    assert executor._stop_task_wait_timeout == 0.0


def test_stop_task_wait_timeout_set_from_download_complete_timeout():
    """`ClientAPILauncherExecutor` must set `_stop_task_wait_timeout = download_complete_timeout`."""
    executor = ClientAPILauncherExecutor(pipe_id="test_pipe", download_complete_timeout=900.0)
    assert executor._stop_task_wait_timeout == 900.0


def test_needs_deferred_stop_launcher_base_returns_false():
    """Launcher base class `needs_deferred_stop()` must return False."""
    from nvflare.app_common.abstract.launcher import Launcher, LauncherRunStatus

    class _MinimalLauncher(Launcher):
        def launch_task(self, task_name, shareable, fl_ctx, abort_signal):
            return True

        def stop_task(self, task_name, fl_ctx, abort_signal):
            pass

        def check_run_status(self, task_name, fl_ctx):
            return LauncherRunStatus.NOT_RUNNING

    assert _MinimalLauncher().needs_deferred_stop() is False


def test_needs_deferred_stop_subprocess_launch_once_false():
    """`SubprocessLauncher(launch_once=False).needs_deferred_stop()` must return True."""
    from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher

    assert SubprocessLauncher(script="echo test", launch_once=False).needs_deferred_stop() is True


def test_needs_deferred_stop_subprocess_launch_once_true():
    """`SubprocessLauncher(launch_once=True).needs_deferred_stop()` must return False."""
    from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher

    assert SubprocessLauncher(script="echo test", launch_once=True).needs_deferred_stop() is False


def test_finalize_captures_deferred_stop_task_name(monkeypatch):
    """`_finalize_external_execution` must capture the task name in `_deferred_stop_task_name`."""
    from unittest.mock import MagicMock

    executor = _make_deferred_stop_executor(monkeypatch)
    executor.launcher = _FakeLauncher(needs_deferred=True, statuses=["success"])
    executor._received_result.set()

    executor._finalize_external_execution("train", MagicMock(), MagicMock(), _FakeAbortSignal())

    assert executor._deferred_stop_task_name == "train"
    executor._deferred_stop_event.wait(timeout=10.0)  # clean up background thread


def test_finalize_deferred_event_cleared_then_set_by_thread(monkeypatch):
    """After `_finalize_external_execution` with needs_deferred=True the event must be
    cleared while the deferred thread is running and set once the thread finishes.

    A gate Event controls when stop_task() proceeds so the test can observe the
    intermediate 'event cleared' state deterministically without any timing races.
    """
    from unittest.mock import MagicMock

    stop_task_entered = threading.Event()  # signals that the deferred thread is inside stop_task
    stop_task_gate = threading.Event()  # test sets this to let stop_task() proceed

    class _GatedLauncher(_FakeLauncher):
        def stop_task(self, task_name, fl_ctx, abort_signal):
            stop_task_entered.set()
            stop_task_gate.wait(timeout=10.0)
            super().stop_task(task_name, fl_ctx, abort_signal)

    executor = _make_deferred_stop_executor(monkeypatch)
    fake_launcher = _GatedLauncher(needs_deferred=True, statuses=["success"])
    executor.launcher = fake_launcher
    executor._received_result.set()

    assert executor._deferred_stop_event.is_set()
    executor._finalize_external_execution("train", MagicMock(), MagicMock(), _FakeAbortSignal())

    # Wait until the deferred thread has entered stop_task() — at this point the
    # finally block has NOT run yet, so the event must still be cleared.
    assert stop_task_entered.wait(timeout=5.0), "Deferred thread did not enter stop_task in time"
    assert not executor._deferred_stop_event.is_set(), "Event must be cleared while deferred thread is in stop_task"

    # Release the gate; the thread completes and the finally block sets the event.
    stop_task_gate.set()
    assert executor._deferred_stop_event.wait(timeout=5.0), "Event must be set after deferred thread completes"
    assert fake_launcher.stop_task_calls == ["train"]


def test_finalize_no_deferred_when_needs_deferred_false(monkeypatch):
    """When `needs_deferred_stop()` is False, `_finalize_external_execution` must
    take the synchronous path and leave the event set."""
    from unittest.mock import MagicMock

    executor = _make_deferred_stop_executor(monkeypatch)
    executor.launcher = _FakeLauncher(needs_deferred=False)
    executor._received_result.set()

    executor._finalize_external_execution("train", MagicMock(), MagicMock(), _FakeAbortSignal())

    assert executor._deferred_stop_event.is_set(), "Synchronous path must not clear the event"


def test_initialize_waits_for_deferred_stop_event(monkeypatch):
    """`_initialize_external_execution` must block on `_deferred_stop_event` until
    the previous round's deferred stop completes, then proceed with launch_task."""
    from unittest.mock import MagicMock

    executor = _make_deferred_stop_executor(monkeypatch)
    executor.launcher = _FakeLauncher(needs_deferred=True)
    executor._deferred_stop_event.clear()  # simulate in-progress deferred stop
    executor._deferred_stop_task_name = "train"

    results = []

    def run_init():
        results.append(executor._initialize_external_execution("train", MagicMock(), MagicMock(), _FakeAbortSignal()))

    t = threading.Thread(target=run_init, daemon=True)
    t.start()

    t.join(timeout=0.3)
    assert t.is_alive(), "_initialize_external_execution should be blocking on the event"

    executor._deferred_stop_event.set()  # simulate deferred stop completing
    t.join(timeout=5.0)
    assert not t.is_alive(), "_initialize_external_execution did not unblock after event was set"
    assert results == [True]


def test_initialize_returns_false_on_abort_during_deferred_wait(monkeypatch):
    """`_initialize_external_execution` must return False immediately when the
    abort signal is triggered while waiting for `_deferred_stop_event`."""
    from unittest.mock import MagicMock

    executor = _make_deferred_stop_executor(monkeypatch)
    executor.launcher = _FakeLauncher(needs_deferred=True)
    executor._deferred_stop_event.clear()
    executor._deferred_stop_task_name = "train"

    result = executor._initialize_external_execution(
        "train", MagicMock(), MagicMock(), _FakeAbortSignal(triggered=True)
    )
    assert result is False


def test_initialize_timeout_fallback_uses_previous_task_name(monkeypatch):
    """On deferred-stop timeout, the fallback `stop_task` call must use
    `_deferred_stop_task_name` (the previous round's task), not the new task name."""
    from unittest.mock import MagicMock, patch

    executor = _make_deferred_stop_executor(monkeypatch, stop_task_wait_timeout=5.0)
    fake_launcher = _FakeLauncher(needs_deferred=True)
    executor.launcher = fake_launcher
    executor._deferred_stop_task_name = "train"  # previous round

    # Replace event with a mock that never becomes set and wait() returns False instantly
    mock_event = MagicMock()
    mock_event.is_set.return_value = False
    mock_event.wait.return_value = False
    executor._deferred_stop_event = mock_event

    # Advance time past deadline after a few calls so the loop exits without sleeping
    base = _real_time.time()
    call_count = [0]

    def fast_time():
        call_count[0] += 1
        return base + (100_000 if call_count[0] > 3 else 0)

    with patch("nvflare.app_common.executors.launcher_executor.time") as mock_time:
        mock_time.time.side_effect = fast_time
        result = executor._initialize_external_execution("validate", MagicMock(), MagicMock(), _FakeAbortSignal())

    assert result is True
    assert fake_launcher.stop_task_calls == [
        "train"
    ], f"Fallback stop_task must use previous task name 'train', got {fake_launcher.stop_task_calls}"


# ---------------------------------------------------------------------------
# launch_once=True scenario: synchronous stop, no deferred thread
# ---------------------------------------------------------------------------


def test_launch_once_finalize_calls_stop_synchronously(monkeypatch):
    """With launch_once=True (needs_deferred=False), stop_task must be called
    synchronously — it appears in stop_task_calls before _finalize returns."""
    from unittest.mock import MagicMock

    executor = _make_deferred_stop_executor(monkeypatch)
    executor.launcher = _FakeLauncher(needs_deferred=False)
    executor._received_result.set()

    executor._finalize_external_execution("train", MagicMock(), MagicMock(), _FakeAbortSignal())

    assert executor.launcher.stop_task_calls == [
        "train"
    ], "launch_once=True: stop_task must be called synchronously during _finalize_external_execution"


def test_launch_once_event_stays_set_after_finalize(monkeypatch):
    """With launch_once=True, _deferred_stop_event must remain set after finalize
    (no background thread is started to clear it)."""
    from unittest.mock import MagicMock

    executor = _make_deferred_stop_executor(monkeypatch)
    executor.launcher = _FakeLauncher(needs_deferred=False)
    executor._received_result.set()

    executor._finalize_external_execution("train", MagicMock(), MagicMock(), _FakeAbortSignal())

    assert (
        executor._deferred_stop_event.is_set()
    ), "launch_once=True: event must remain set — synchronous stop does not clear it"


def test_launch_once_sequential_rounds_no_blocking(monkeypatch):
    """With launch_once=True, _initialize_external_execution for round N+1 must not
    block on the deferred stop event (which is never cleared in this path)."""
    from unittest.mock import MagicMock

    executor = _make_deferred_stop_executor(monkeypatch)
    executor.launcher = _FakeLauncher(needs_deferred=False)
    executor._received_result.set()

    # Finalize round N
    executor._finalize_external_execution("train", MagicMock(), MagicMock(), _FakeAbortSignal())

    # Initialize round N+1 must return True immediately (event is set, no wait)
    result = executor._initialize_external_execution("train", MagicMock(), MagicMock(), _FakeAbortSignal())
    assert result is True


def test_launch_once_deferred_task_name_not_captured(monkeypatch):
    """With needs_deferred=False, _deferred_stop_task_name must NOT be updated —
    task name capture only happens in the deferred branch."""
    from unittest.mock import MagicMock

    executor = _make_deferred_stop_executor(monkeypatch)
    executor.launcher = _FakeLauncher(needs_deferred=False)
    executor._received_result.set()

    executor._finalize_external_execution("train", MagicMock(), MagicMock(), _FakeAbortSignal())

    assert (
        executor._deferred_stop_task_name == ""
    ), "launch_once=True: _deferred_stop_task_name must not be updated in the synchronous path"


# ---------------------------------------------------------------------------
# Swarm learning scenario: launch_once=True, multiple task types in sequence
# ---------------------------------------------------------------------------


def test_swarm_each_task_type_stop_called_synchronously(monkeypatch):
    """In a swarm job (launch_once=True), stop_task must be called synchronously for
    each of train / validate / submit_model in the order they are finalized."""
    from unittest.mock import MagicMock

    executor = _make_deferred_stop_executor(monkeypatch)
    executor.launcher = _FakeLauncher(needs_deferred=False)

    for task_name in ("train", "validate", "submit_model"):
        executor._received_result.set()
        executor._finalize_external_execution(task_name, MagicMock(), MagicMock(), _FakeAbortSignal())

    assert executor.launcher.stop_task_calls == [
        "train",
        "validate",
        "submit_model",
    ], "Swarm: stop_task must be called for each task type in sequence"


def test_swarm_event_stays_set_across_all_task_types(monkeypatch):
    """In a swarm job, _deferred_stop_event must remain set after every task type —
    no deferred threads are ever started with launch_once=True."""
    from unittest.mock import MagicMock

    executor = _make_deferred_stop_executor(monkeypatch)
    executor.launcher = _FakeLauncher(needs_deferred=False)

    for task_name in ("train", "validate", "submit_model"):
        executor._received_result.set()
        executor._finalize_external_execution(task_name, MagicMock(), MagicMock(), _FakeAbortSignal())
        assert (
            executor._deferred_stop_event.is_set()
        ), f"Swarm: event must remain set after finalizing task '{task_name}'"


def test_swarm_initialize_never_blocks_on_event(monkeypatch):
    """In a swarm job, _initialize_external_execution must never block on the
    deferred stop event because it is never cleared by a needs_deferred=False launcher."""
    from unittest.mock import MagicMock

    executor = _make_deferred_stop_executor(monkeypatch)
    executor.launcher = _FakeLauncher(needs_deferred=False)

    for task_name in ("train", "validate", "submit_model"):
        result = executor._initialize_external_execution(task_name, MagicMock(), MagicMock(), _FakeAbortSignal())
        assert result is True, f"Swarm: _initialize_external_execution must succeed for task '{task_name}'"
        assert (
            executor._deferred_stop_event.is_set()
        ), f"Swarm: event must remain set before/after initialize for task '{task_name}'"


# ---------------------------------------------------------------------------
# _executing guard: prevents BEFORE_TASK_EXECUTION from replacing
# the pipe handler mid-transaction (Swarm deadlock fix)
# ---------------------------------------------------------------------------


class _FakePipeHandler:
    """Minimal PipeHandler stub that records stop() calls and provides get_next()."""

    def __init__(self, identity="default"):
        self.identity = identity
        self.stopped = False
        self.asked_to_stop = False
        self._replies = []

    def stop(self, close_pipe=True):
        self.stopped = True

    def start(self):
        pass

    def send_to_peer(self, msg, timeout=None, abort_signal=None):
        return True

    def get_next(self):
        if self._replies:
            return self._replies.pop(0)
        return None

    def notify_abort(self, task_id):
        pass

    def set_status_cb(self, cb):
        pass


def test_executing_flag_starts_unset():
    """_executing must start as an unset threading.Event."""
    executor = TaskExchanger(pipe_id="test_pipe")
    assert not executor._executing.is_set()


def test_handle_event_skips_reset_when_executing(monkeypatch):
    """BEFORE_TASK_EXECUTION must not replace the pipe handler when _executing is set."""
    from unittest.mock import MagicMock

    executor = TaskExchanger(pipe_id="test_pipe")
    monkeypatch.setattr(TaskExchanger, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(TaskExchanger, "log_debug", lambda self, fl_ctx, msg: None)

    original_handler = _FakePipeHandler(identity="original")
    executor.pipe_handler = original_handler

    executor._executing.set()
    executor.handle_event(EventType.BEFORE_TASK_EXECUTION, MagicMock())

    assert executor.pipe_handler is original_handler, "Handler must not be replaced while _executing is set"
    assert not original_handler.stopped, "Original handler must not be stopped while _executing is set"


def test_handle_event_resets_handler_when_not_executing(monkeypatch):
    """BEFORE_TASK_EXECUTION must replace the pipe handler when _executing is not set."""
    from unittest.mock import MagicMock

    from nvflare.fuel.utils.pipe.pipe import Pipe

    executor = TaskExchanger(pipe_id="test_pipe")
    monkeypatch.setattr(TaskExchanger, "log_info", lambda self, fl_ctx, msg: None)

    old_handler = _FakePipeHandler(identity="old")
    executor.pipe_handler = old_handler
    executor.pipe = MagicMock(spec=Pipe)

    assert not executor._executing.is_set()
    executor.handle_event(EventType.BEFORE_TASK_EXECUTION, MagicMock())

    assert old_handler.stopped, "Old handler must be stopped"
    assert executor.pipe_handler is not old_handler, "Handler must be replaced when _executing is not set"


def test_task_exchanger_execute_acquires_and_releases_flag(monkeypatch):
    """TaskExchanger.execute() must set _executing during execution and clear it after."""
    from unittest.mock import MagicMock

    from nvflare.apis.shareable import Shareable

    executor = TaskExchanger(pipe_id="test_pipe")
    monkeypatch.setattr(TaskExchanger, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(TaskExchanger, "log_debug", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(TaskExchanger, "log_error", lambda self, fl_ctx, msg: None)

    observed_during = []

    def _spy_do_execute(self, task_name, shareable, fl_ctx, abort_signal):
        observed_during.append(self._executing.is_set())
        return make_reply(ReturnCode.BAD_TASK_DATA)

    monkeypatch.setattr(TaskExchanger, "_do_execute", _spy_do_execute)

    shareable = Shareable()
    fl_ctx = MagicMock()
    fl_ctx.get_job_id.return_value = "j1"
    fl_ctx.get_identity_name.return_value = "site1"
    abort_signal = _FakeAbortSignal()

    assert not executor._executing.is_set()
    executor.execute("train", shareable, fl_ctx, abort_signal)
    assert not executor._executing.is_set(), "_executing must be cleared after execute() returns"
    assert observed_during == [True], "_executing must be set during _do_execute"


def test_task_exchanger_execute_clears_flag_on_exception(monkeypatch):
    """_executing must be cleared even if _do_execute raises an exception."""
    from unittest.mock import MagicMock

    from nvflare.apis.shareable import Shareable

    executor = TaskExchanger(pipe_id="test_pipe")
    monkeypatch.setattr(TaskExchanger, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(TaskExchanger, "log_debug", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(TaskExchanger, "log_error", lambda self, fl_ctx, msg: None)

    def _exploding_execute(self, task_name, shareable, fl_ctx, abort_signal):
        raise RuntimeError("boom")

    monkeypatch.setattr(TaskExchanger, "_do_execute", _exploding_execute)

    with pytest.raises(RuntimeError, match="boom"):
        executor.execute("train", Shareable(), MagicMock(), _FakeAbortSignal())

    assert not executor._executing.is_set(), "_executing must be cleared even after exception"


def test_launcher_executor_sets_flag_before_initialize(monkeypatch):
    """LauncherExecutor.execute() must set _executing before _initialize_external_execution."""
    from unittest.mock import MagicMock

    from nvflare.apis.shareable import Shareable

    observed_in_init = []

    executor = _make_deferred_stop_executor(monkeypatch)
    executor.launcher = _FakeLauncher(needs_deferred=False)

    def _spy_init(self, task_name, shareable, fl_ctx, abort_signal):
        observed_in_init.append(self._executing.is_set())
        return False

    monkeypatch.setattr(LauncherExecutor, "_initialize_external_execution", _spy_init)

    executor.execute("train", Shareable(), MagicMock(), _FakeAbortSignal())

    assert observed_in_init == [True], "_executing must be set before _initialize_external_execution runs"
    assert not executor._executing.is_set(), "_executing must be cleared after execute() returns"


def test_launcher_executor_clears_flag_after_full_lifecycle(monkeypatch):
    """_executing must be cleared after the full LauncherExecutor.execute() lifecycle
    (init + super().execute() + finalize)."""
    from unittest.mock import MagicMock

    from nvflare.apis.shareable import Shareable

    executor = _make_deferred_stop_executor(monkeypatch)
    executor.launcher = _FakeLauncher(needs_deferred=False)

    result_shareable = Shareable()
    result_shareable.set_return_code(ReturnCode.OK)

    def _fake_do_execute(self, task_name, shareable, fl_ctx, abort_signal):
        return result_shareable

    monkeypatch.setattr(TaskExchanger, "_do_execute", _fake_do_execute)
    monkeypatch.setattr(LauncherExecutor, "_finalize_external_execution", lambda self, tn, sh, fl, ab: True)

    executor.execute("train", Shareable(), MagicMock(), _FakeAbortSignal())

    assert not executor._executing.is_set(), "_executing must be cleared after full lifecycle"


def test_ownership_super_execute_does_not_clear_flag(monkeypatch):
    """When LauncherExecutor sets _executing, super().execute()'s finally block must NOT
    clear it (acquired=False). The flag must remain set until LauncherExecutor's finally."""
    from unittest.mock import MagicMock

    from nvflare.apis.shareable import Shareable

    flag_after_super = []

    executor = _make_deferred_stop_executor(monkeypatch)
    executor.launcher = _FakeLauncher(needs_deferred=False)

    result_shareable = Shareable()
    result_shareable.set_return_code(ReturnCode.OK)

    def _fake_do_execute(self, task_name, shareable, fl_ctx, abort_signal):
        return result_shareable

    monkeypatch.setattr(TaskExchanger, "_do_execute", _fake_do_execute)

    def _spy_finalize(self, task_name, shareable, fl_ctx, abort_signal):
        flag_after_super.append(self._executing.is_set())
        return True

    monkeypatch.setattr(LauncherExecutor, "_finalize_external_execution", _spy_finalize)

    executor.execute("train", Shareable(), MagicMock(), _FakeAbortSignal())

    assert flag_after_super == [True], (
        "_executing must still be set during _finalize_external_execution "
        "(super().execute()'s finally must not clear it)"
    )


def test_concurrent_before_task_execution_blocked_during_initialize(monkeypatch):
    """A BEFORE_TASK_EXECUTION event arriving on another thread during
    _initialize_external_execution must be blocked by the _executing guard."""
    from unittest.mock import MagicMock

    from nvflare.apis.shareable import Shareable

    executor = _make_deferred_stop_executor(monkeypatch)
    executor.launcher = _FakeLauncher(needs_deferred=False)

    original_handler = _FakePipeHandler(identity="original")
    executor.pipe_handler = original_handler
    executor.pipe = MagicMock()

    init_entered = threading.Event()
    init_gate = threading.Event()
    handler_replaced = []

    def _blocking_init(self, task_name, shareable, fl_ctx, abort_signal):
        init_entered.set()
        init_gate.wait(timeout=10.0)
        return False

    monkeypatch.setattr(LauncherExecutor, "_initialize_external_execution", _blocking_init)

    def run_execute():
        executor.execute("train", Shareable(), MagicMock(), _FakeAbortSignal())

    t = threading.Thread(target=run_execute, daemon=True)
    t.start()

    assert init_entered.wait(timeout=5.0), "_initialize_external_execution did not start"

    handler_before = executor.pipe_handler
    executor.handle_event(EventType.BEFORE_TASK_EXECUTION, MagicMock())
    handler_after = executor.pipe_handler

    handler_replaced.append(handler_before is not handler_after)

    init_gate.set()
    t.join(timeout=5.0)

    assert handler_replaced == [
        False
    ], "BEFORE_TASK_EXECUTION must NOT replace the handler during _initialize_external_execution"

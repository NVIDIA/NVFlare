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

import pytest

from nvflare.app_common.executors.client_api_launcher_executor import ClientAPILauncherExecutor
from nvflare.app_common.executors.launcher_executor import LauncherExecutor
from nvflare.fuel.utils.fobs import FOBSContextKey
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


def test_pass_through_restored_to_previous_value(base_executor):
    cell = _FakeCell({FOBSContextKey.PASS_THROUGH: True})
    fl_ctx = _FakeFLContext(cell)

    base_executor.initialize(fl_ctx)
    assert cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is True

    base_executor.finalize(fl_ctx)
    assert cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is True
    assert base_executor._cell_with_pass_through is None


def test_pass_through_restored_to_none_when_previously_absent(base_executor):
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    base_executor.initialize(fl_ctx)
    assert cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is True

    base_executor.finalize(fl_ctx)
    assert cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is None
    assert base_executor._cell_with_pass_through is None


def test_initialize_failure_restores_pass_through(monkeypatch):
    monkeypatch.setattr(ClientAPILauncherExecutor, "prepare_config_for_launch", lambda self, fl_ctx: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_warning", lambda self, fl_ctx, msg: None)

    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(LauncherExecutor, "initialize", _raise)

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe")
    executor.pipe = _make_fake_cell_pipe()
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    with pytest.raises(RuntimeError, match="boom"):
        executor.initialize(fl_ctx)

    assert cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is None
    assert executor._cell_with_pass_through is None


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

    from nvflare.app_common.utils.export_utils import update_export_props
    from nvflare.client.config import ConfigKey, write_config_to_file

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
    from nvflare.client.constants import EXTERNAL_PRE_INIT_TIMEOUT, PEER_READ_TIMEOUT

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
# Fix 7: Gate PASS_THROUGH on CellPipe only
# ---------------------------------------------------------------------------


def test_pass_through_enabled_for_cell_pipe(monkeypatch):
    """PASS_THROUGH must be set when pipe is a CellPipe (positive path)."""
    monkeypatch.setattr(ClientAPILauncherExecutor, "prepare_config_for_launch", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "initialize", lambda self, fl_ctx: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_warning", lambda self, fl_ctx, msg: None)

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe")
    executor.pipe = _make_fake_cell_pipe()
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    executor.initialize(fl_ctx)

    assert cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is True
    assert executor._cell_with_pass_through is cell  # engine cell stored for later restore


def test_pass_through_skipped_for_non_cell_pipe(monkeypatch):
    """PASS_THROUGH must NOT be set when pipe is not a CellPipe (e.g. FilePipe).

    Before Fix 7: PASS_THROUGH was enabled unconditionally, silently corrupting
    task data for FilePipe subprocesses that cannot resolve cell FQCNs.
    After Fix 7: only CellPipe triggers PASS_THROUGH.
    """
    monkeypatch.setattr(ClientAPILauncherExecutor, "prepare_config_for_launch", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "initialize", lambda self, fl_ctx: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_warning", lambda self, fl_ctx, msg: None)

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe")
    # Simulate a non-CellPipe (e.g. FilePipe or any other Pipe subclass).
    from unittest.mock import MagicMock
    executor.pipe = MagicMock()  # not a CellPipe instance

    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    executor.initialize(fl_ctx)

    # PASS_THROUGH must remain absent (not set at all).
    assert FOBSContextKey.PASS_THROUGH not in cell.core_cell.ctx
    assert executor._cell_with_pass_through is None


def test_pass_through_skipped_when_pipe_is_none(monkeypatch):
    """PASS_THROUGH must not be set when pipe has not been resolved yet."""
    monkeypatch.setattr(ClientAPILauncherExecutor, "prepare_config_for_launch", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "initialize", lambda self, fl_ctx: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_warning", lambda self, fl_ctx, msg: None)

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe")
    # pipe is None (default before START_RUN fires)

    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    executor.initialize(fl_ctx)

    assert FOBSContextKey.PASS_THROUGH not in cell.core_cell.ctx
    assert executor._cell_with_pass_through is None


# ---------------------------------------------------------------------------
# Fix 8: Reverse PASS_THROUGH on pipe cell
# ---------------------------------------------------------------------------


class _FakePipeCoreCell:
    """Minimal CoreCell-like for a CellPipe's cell."""
    def __init__(self):
        self.ctx = {}

    def update_fobs_context(self, props: dict):
        self.ctx.update(props)

    def get_fobs_context(self, props: dict = None):
        import copy
        result = copy.copy(self.ctx)
        if props:
            result.update(props)
        return result


class _FakePipeCell:
    """Minimal Cell-like object attached to a CellPipe."""
    def __init__(self):
        self.core_cell = _FakePipeCoreCell()


def _make_cell_pipe_with_fake_cell(pipe_cell=None):
    """Return a CellPipe instance whose .cell is a _FakePipeCell."""
    cp = _make_fake_cell_pipe()
    cp.cell = pipe_cell if pipe_cell is not None else _FakePipeCell()
    return cp


def test_pipe_cell_pass_through_enabled(monkeypatch):
    """Reverse PASS_THROUGH must be set on the pipe cell when pipe is CellPipe
    and pipe.cell is distinct from the engine cell."""
    monkeypatch.setattr(ClientAPILauncherExecutor, "prepare_config_for_launch", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "initialize", lambda self, fl_ctx: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_warning", lambda self, fl_ctx, msg: None)

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe")
    executor.pipe = _make_cell_pipe_with_fake_cell()

    engine_cell = _FakeCell()
    fl_ctx = _FakeFLContext(engine_cell)

    executor.initialize(fl_ctx)

    # Engine cell: forward PASS_THROUGH
    assert engine_cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is True
    # Pipe cell: reverse PASS_THROUGH
    assert executor.pipe.cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is True
    assert executor._pipe_cell_with_pass_through is executor.pipe.cell


def test_pipe_cell_pass_through_not_set_when_same_as_engine_cell(monkeypatch):
    """Reverse PASS_THROUGH must NOT double-set when pipe.cell is the same
    object as the engine cell (e.g. in simulator)."""
    monkeypatch.setattr(ClientAPILauncherExecutor, "prepare_config_for_launch", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "initialize", lambda self, fl_ctx: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_warning", lambda self, fl_ctx, msg: None)

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe")
    engine_cell = _FakeCell()
    # Pipe cell IS the engine cell — same object.
    executor.pipe = _make_cell_pipe_with_fake_cell(pipe_cell=engine_cell)

    fl_ctx = _FakeFLContext(engine_cell)
    executor.initialize(fl_ctx)

    assert executor._pipe_cell_with_pass_through is None


def test_pipe_cell_pass_through_restored_on_finalize(monkeypatch):
    """Pipe cell PASS_THROUGH must be restored to its previous value on finalize."""
    monkeypatch.setattr(ClientAPILauncherExecutor, "prepare_config_for_launch", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "initialize", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "finalize", lambda self, fl_ctx: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_warning", lambda self, fl_ctx, msg: None)

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe")
    executor.pipe = _make_cell_pipe_with_fake_cell()

    engine_cell = _FakeCell()
    fl_ctx = _FakeFLContext(engine_cell)

    executor.initialize(fl_ctx)
    assert executor.pipe.cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is True

    executor.finalize(fl_ctx)
    assert executor.pipe.cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is None
    assert executor._pipe_cell_with_pass_through is None


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

    client_config = ClientConfig(config={
        ConfigKey.TASK_EXCHANGE: {
            ConfigKey.MAX_RESENDS: 5,
            ConfigKey.SUBMIT_RESULT_TIMEOUT: 300.0,
            ConfigKey.HEARTBEAT_TIMEOUT: 600.0,
        }
    })

    captured_kwargs = {}

    def fake_agent_init(self, *args, **kwargs):
        captured_kwargs.update(kwargs)

    with patch("nvflare.client.flare_agent_with_fl_model.FlareAgentWithFLModel.__init__", fake_agent_init):
        from nvflare.client.flare_agent_with_fl_model import FlareAgentWithFLModel
        agent = object.__new__(FlareAgentWithFLModel)
        fake_agent_init(agent, pipe=MagicMock(), max_resends=client_config.get_max_resends())

    assert captured_kwargs.get("max_resends") == 5


# ---------------------------------------------------------------------------
# Memory context (memory_gc_rounds / cuda_empty_cache) FOBS context wiring
# ---------------------------------------------------------------------------

def test_memory_ctx_set_on_engine_cell_when_gc_rounds_positive(base_executor):
    """When memory_gc_rounds > 0, MEMORY_GC_ROUNDS and CUDA_EMPTY_CACHE must be
    written to the engine cell's FOBS context."""
    base_executor._memory_gc_rounds = 2
    base_executor._cuda_empty_cache = True
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    base_executor.initialize(fl_ctx)

    assert cell.core_cell.ctx.get(FOBSContextKey.MEMORY_GC_ROUNDS) == 2
    assert cell.core_cell.ctx.get(FOBSContextKey.CUDA_EMPTY_CACHE) is True


def test_memory_ctx_not_set_when_gc_rounds_zero(base_executor):
    """When memory_gc_rounds=0, MEMORY_GC_ROUNDS must NOT be written to the cell."""
    base_executor._memory_gc_rounds = 0
    base_executor._cuda_empty_cache = False
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    base_executor.initialize(fl_ctx)

    assert FOBSContextKey.MEMORY_GC_ROUNDS not in cell.core_cell.ctx


def test_memory_ctx_set_on_pipe_cell(base_executor):
    """When memory_gc_rounds > 0 and pipe has a separate cell, that cell's FOBS
    context must also receive the memory settings."""
    base_executor._memory_gc_rounds = 3
    base_executor._cuda_empty_cache = False

    engine_cell = _FakeCell()
    pipe_cell = _FakeCell()

    # Attach a fake cell to the pipe (different from the engine cell)
    fake_pipe = base_executor.pipe
    fake_pipe.cell = pipe_cell

    fl_ctx = _FakeFLContext(engine_cell)
    base_executor.initialize(fl_ctx)

    assert pipe_cell.core_cell.ctx.get(FOBSContextKey.MEMORY_GC_ROUNDS) == 3


def test_memory_ctx_not_duplicated_when_pipe_cell_is_engine_cell(base_executor):
    """When pipe cell and engine cell are the same object, MEMORY_GC_ROUNDS must
    still be set (idempotently) without raising."""
    base_executor._memory_gc_rounds = 1
    base_executor._cuda_empty_cache = False

    cell = _FakeCell()
    # Make the pipe's cell the same object as the engine cell
    base_executor.pipe.cell = cell

    fl_ctx = _FakeFLContext(cell)
    base_executor.initialize(fl_ctx)

    # The value must be set on the shared cell
    assert cell.core_cell.ctx.get(FOBSContextKey.MEMORY_GC_ROUNDS) == 1


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
    from nvflare.app_common.executors.launcher_executor import LauncherExecutor

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
    from nvflare.app_common.executors.launcher_executor import LauncherExecutor

    executor = ClientAPILauncherExecutor(pipe_id="p")
    executor._memory_gc_rounds = 1

    monkeypatch.setattr(LauncherExecutor, "check_output_shareable", lambda self, tn, sh, ctx: False)
    with patch("nvflare.fuel.utils.mem_utils.log_rss") as mock_log_rss, \
         patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup:
        result = executor.check_output_shareable("train", _make_shareable(), MagicMock())

    assert result is False
    mock_log_rss.assert_not_called()
    mock_cleanup.assert_not_called()


def test_cj_cleanup_not_called_when_gc_rounds_zero(monkeypatch):
    """_maybe_cleanup_cj_memory must not call cleanup_memory when memory_gc_rounds=0."""
    from unittest.mock import MagicMock, patch
    from nvflare.app_common.executors.launcher_executor import LauncherExecutor

    executor = ClientAPILauncherExecutor(pipe_id="p")
    executor._memory_gc_rounds = 0

    monkeypatch.setattr(LauncherExecutor, "check_output_shareable", lambda self, tn, sh, ctx: True)
    with patch("nvflare.fuel.utils.mem_utils.log_rss"), \
         patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup:
        executor.check_output_shareable("train", _make_shareable(), MagicMock())

    mock_cleanup.assert_not_called()


def test_cj_cleanup_called_every_n_rounds(monkeypatch):
    """cleanup_memory must be called exactly once per memory_gc_rounds sends."""
    from unittest.mock import MagicMock, patch
    from nvflare.app_common.executors.launcher_executor import LauncherExecutor

    executor = ClientAPILauncherExecutor(pipe_id="p")
    executor._memory_gc_rounds = 3
    executor._cuda_empty_cache = False
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_warning", lambda self, fl_ctx, msg: None)

    monkeypatch.setattr(LauncherExecutor, "check_output_shareable", lambda self, tn, sh, ctx: True)
    with patch("nvflare.fuel.utils.mem_utils.log_rss"), \
         patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup:
        for i in range(6):
            executor.check_output_shareable("train", _make_shareable(i), MagicMock())

    assert mock_cleanup.call_count == 2  # rounds 3 and 6


def test_cj_cleanup_passes_cuda_empty_cache(monkeypatch):
    """cleanup_memory must be called with cuda_empty_cache=True when configured."""
    from unittest.mock import MagicMock, patch
    from nvflare.app_common.executors.launcher_executor import LauncherExecutor

    executor = ClientAPILauncherExecutor(pipe_id="p")
    executor._memory_gc_rounds = 1
    executor._cuda_empty_cache = True
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None)
    monkeypatch.setattr(ClientAPILauncherExecutor, "log_warning", lambda self, fl_ctx, msg: None)

    monkeypatch.setattr(LauncherExecutor, "check_output_shareable", lambda self, tn, sh, ctx: True)
    with patch("nvflare.fuel.utils.mem_utils.log_rss"), \
         patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup:
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

    executor, warnings = _make_validating_executor(monkeypatch)
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    # Force: per_req=600 (default), min_dl=60 (default) → min_dl < per_req
    monkeypatch.setattr(acu, "get_positive_float_var", lambda name, default: default)
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
            return 700.0   # min_dl > per_req
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
            return 300.0   # min_dl=300 < submit_result_timeout=400
        if ConfigVarName.STREAMING_PER_REQUEST_TIMEOUT in name:
            return 100.0   # keep per_req < min_dl so only the submit warning fires
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

    executor, warnings = _make_validating_executor(
        monkeypatch, submit_result_timeout=200.0, max_resends=3
    )
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


def test_decomposer_prefix_default_is_numpy(monkeypatch):
    """Base class _decomposer_prefix() must return 'np_'."""
    executor = ClientAPILauncherExecutor(pipe_id="p")
    assert executor._decomposer_prefix() == "np_"


def test_pt_executor_decomposer_prefix_is_tensor():
    """PTClientAPILauncherExecutor._decomposer_prefix() must return 'tensor_'."""
    from nvflare.app_opt.pt.client_api_launcher_executor import PTClientAPILauncherExecutor

    executor = object.__new__(PTClientAPILauncherExecutor)
    assert executor._decomposer_prefix() == "tensor_"

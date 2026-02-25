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

import nvflare.app_common.executors.client_api_launcher_executor as client_api_launcher_executor_module
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api_launcher_executor import ClientAPILauncherExecutor
from nvflare.app_common.executors.launcher_executor import LauncherExecutor
from nvflare.fuel.utils.fobs import FOBSContextKey


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

    def get_peer_context(self):
        return self._peer_ctx


class _FakeCellPipe:
    def __init__(self, cell):
        self.cell = cell


@pytest.fixture
def base_executor(monkeypatch):
    monkeypatch.setattr(ClientAPILauncherExecutor, "prepare_config_for_launch", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "initialize", lambda self, fl_ctx: None)
    monkeypatch.setattr(LauncherExecutor, "finalize", lambda self, fl_ctx: None)
    return ClientAPILauncherExecutor(pipe_id="test_pipe")


def test_pass_through_restored_to_previous_value(base_executor):
    cell = _FakeCell({FOBSContextKey.PASS_THROUGH: True})
    fl_ctx = _FakeFLContext(cell)

    base_executor.initialize(fl_ctx)
    assert cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is True

    base_executor.finalize(fl_ctx)
    assert cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is True
    assert base_executor._cell_with_pass_through is None
    assert base_executor._pipe_cell_with_pass_through is None


def test_pass_through_restored_to_none_when_previously_absent(base_executor):
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    base_executor.initialize(fl_ctx)
    assert cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is True

    base_executor.finalize(fl_ctx)
    assert cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is None
    assert base_executor._cell_with_pass_through is None
    assert base_executor._pipe_cell_with_pass_through is None


def test_pipe_cell_pass_through_restored_to_previous_value(base_executor, monkeypatch):
    monkeypatch.setattr(client_api_launcher_executor_module, "CellPipe", _FakeCellPipe)
    engine_cell = _FakeCell()
    pipe_cell = _FakeCell({FOBSContextKey.PASS_THROUGH: True})
    base_executor.pipe = _FakeCellPipe(pipe_cell)
    fl_ctx = _FakeFLContext(engine_cell)

    base_executor.initialize(fl_ctx)
    assert engine_cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is True
    assert pipe_cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is True

    base_executor.finalize(fl_ctx)
    assert engine_cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is None
    assert pipe_cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is True
    assert base_executor._cell_with_pass_through is None
    assert base_executor._pipe_cell_with_pass_through is None


def test_initialize_failure_restores_pass_through(monkeypatch):
    monkeypatch.setattr(ClientAPILauncherExecutor, "prepare_config_for_launch", lambda self, fl_ctx: None)

    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(LauncherExecutor, "initialize", _raise)

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe")
    cell = _FakeCell()
    fl_ctx = _FakeFLContext(cell)

    with pytest.raises(RuntimeError, match="boom"):
        executor.initialize(fl_ctx)

    assert cell.core_cell.ctx[FOBSContextKey.PASS_THROUGH] is None
    assert executor._cell_with_pass_through is None
    assert executor._pipe_cell_with_pass_through is None


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


def test_cj_memory_cleanup_runs_on_interval(monkeypatch):
    monkeypatch.setattr(LauncherExecutor, "check_output_shareable", lambda self, task_name, shareable, fl_ctx: True)

    cleanup_calls = []
    monkeypatch.setattr(
        client_api_launcher_executor_module,
        "cleanup_memory",
        lambda cuda_empty_cache: cleanup_calls.append(cuda_empty_cache),
    )

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe", memory_gc_rounds=2, cuda_empty_cache=True)
    fl_ctx = _FakeFLContext(_FakeCell())
    shareable = Shareable()
    shareable.set_header(AppConstants.CURRENT_ROUND, 1)

    assert executor.check_output_shareable("train", shareable, fl_ctx) is True
    assert cleanup_calls == []

    assert executor.check_output_shareable("train", shareable, fl_ctx) is True
    assert cleanup_calls == [True]

    assert executor.check_output_shareable("train", shareable, fl_ctx) is True
    assert cleanup_calls == [True]


def test_cj_memory_profile_logs_rss(monkeypatch):
    monkeypatch.setattr(LauncherExecutor, "check_output_shareable", lambda self, task_name, shareable, fl_ctx: True)
    tags = []
    monkeypatch.setattr(client_api_launcher_executor_module, "log_rss", lambda tag: tags.append(tag))

    executor = ClientAPILauncherExecutor(pipe_id="test_pipe")
    fl_ctx = _FakeFLContext(_FakeCell())
    shareable = Shareable()

    assert executor.check_output_shareable("train", shareable, fl_ctx) is True
    assert any("after_send" in t for t in tags)

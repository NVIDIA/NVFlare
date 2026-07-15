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
import threading
from unittest.mock import MagicMock, patch

import pytest

from nvflare.client import api as flare_api
from nvflare.client.api_context import APIContext, ClientAPIType
from nvflare.client.api_spec import CLIENT_API_TYPE_KEY
from nvflare.client.cell.bootstrap import BOOTSTRAP_SCHEMA_VERSION, EXTERNAL_PROCESS_EXECUTION_MODE, BootstrapKey
from nvflare.client.in_process.api import InProcessClientAPI


def _write_config(path, config):
    path.write_text(json.dumps(config))
    return str(path)


def _typed_bootstrap():
    return {
        BootstrapKey.SCHEMA_VERSION: BOOTSTRAP_SCHEMA_VERSION,
        BootstrapKey.EXECUTION_MODE: EXTERNAL_PROCESS_EXECUTION_MODE,
    }


@pytest.fixture(autouse=True)
def _clean_global_contexts():
    old_contexts = flare_api.context_dict
    old_default = flare_api.default_context
    old_runtime_shutdown = flare_api._runtime_shutdown
    had_thread_context = hasattr(flare_api._thread_context, "context")
    old_thread_context = getattr(flare_api._thread_context, "context", None)
    flare_api.context_dict = {}
    flare_api.default_context = None
    flare_api._runtime_shutdown = False
    if had_thread_context:
        del flare_api._thread_context.context
    yield
    flare_api.context_dict = old_contexts
    flare_api.default_context = old_default
    flare_api._runtime_shutdown = old_runtime_shutdown
    if had_thread_context:
        flare_api._thread_context.context = old_thread_context
    elif hasattr(flare_api._thread_context, "context"):
        del flare_api._thread_context.context


class TestAPIContextSelection:
    def test_explicit_typed_bootstrap_selects_cell_and_passes_exact_path(self, tmp_path, monkeypatch):
        config_file = _write_config(tmp_path / "attach-bootstrap.json", _typed_bootstrap())
        monkeypatch.delenv(CLIENT_API_TYPE_KEY, raising=False)

        with patch("nvflare.client.cell.api.CellClientAPI") as cell_api_cls:
            ctx = APIContext(rank="0", config_file=config_file)

        cell_api_cls.assert_called_once_with(bootstrap_file=config_file)
        cell_api_cls.return_value.init.assert_called_once_with(rank="0")
        assert ctx.api_type == ClientAPIType.CELL_API
        assert ctx.api is cell_api_cls.return_value

    def test_env_cell_selection_without_explicit_config_keeps_bootstrap_env_behavior(self, monkeypatch):
        monkeypatch.setenv(CLIENT_API_TYPE_KEY, ClientAPIType.CELL_API.value)

        with patch("nvflare.client.cell.api.CellClientAPI") as cell_api_cls:
            ctx = APIContext(rank="0")

        cell_api_cls.assert_called_once_with()
        assert ctx.api_type == ClientAPIType.CELL_API

    def test_legacy_ex_process_config_still_uses_env_selection(self, tmp_path, monkeypatch):
        config_file = _write_config(tmp_path / "legacy.json", {"TASK_EXCHANGE": {}})
        monkeypatch.setenv(CLIENT_API_TYPE_KEY, ClientAPIType.EX_PROCESS_API.value)

        with patch("nvflare.client.api_context.ExProcessClientAPI") as ex_process_api_cls:
            ctx = APIContext(rank="0", config_file=config_file)

        ex_process_api_cls.assert_called_once_with(config_file=config_file)
        ex_process_api_cls.return_value.init.assert_called_once_with(rank="0")
        assert ctx.api_type == ClientAPIType.EX_PROCESS_API

    def test_untyped_explicit_config_preserves_default_in_process_selection(self, tmp_path, monkeypatch):
        config_file = _write_config(tmp_path / "legacy.json", {"TASK_EXCHANGE": {}})
        monkeypatch.delenv(CLIENT_API_TYPE_KEY, raising=False)
        in_process_api = MagicMock(spec=InProcessClientAPI)

        with patch("nvflare.client.api_context.data_bus.get_data", return_value=in_process_api):
            ctx = APIContext(rank="0", config_file=config_file)

        assert ctx.api_type == ClientAPIType.IN_PROCESS_API
        assert ctx.api is in_process_api
        in_process_api.init.assert_called_once_with(rank="0")

    def test_owner_closed_in_process_api_marks_context_shutdown(self, monkeypatch):
        monkeypatch.delenv(CLIENT_API_TYPE_KEY, raising=False)
        in_process_api = MagicMock(spec=InProcessClientAPI)
        in_process_api.closed = False

        with patch("nvflare.client.api_context.data_bus.get_data", return_value=in_process_api):
            ctx = APIContext(rank="0")

        assert ctx.is_shutdown is False
        in_process_api.closed = True
        assert ctx.is_shutdown is True

    @pytest.mark.parametrize(
        "config,match",
        [
            (
                {
                    BootstrapKey.SCHEMA_VERSION: BOOTSTRAP_SCHEMA_VERSION + 1,
                    BootstrapKey.EXECUTION_MODE: EXTERNAL_PROCESS_EXECUTION_MODE,
                },
                "unsupported Client API bootstrap schema_version",
            ),
            (
                {
                    BootstrapKey.SCHEMA_VERSION: BOOTSTRAP_SCHEMA_VERSION,
                    BootstrapKey.EXECUTION_MODE: "attach",
                },
                "unsupported Client API bootstrap execution_mode",
            ),
            (
                {BootstrapKey.SCHEMA_VERSION: BOOTSTRAP_SCHEMA_VERSION},
                "missing required field 'execution_mode'",
            ),
        ],
    )
    def test_explicit_typed_bootstrap_rejects_unsupported_or_partial_envelope(
        self, tmp_path, monkeypatch, config, match
    ):
        config_file = _write_config(tmp_path / "bootstrap.json", config)
        monkeypatch.delenv(CLIENT_API_TYPE_KEY, raising=False)

        with pytest.raises(ValueError, match=match):
            APIContext(config_file=config_file)

    def test_typed_bootstrap_rejects_conflicting_env_selection(self, tmp_path, monkeypatch):
        config_file = _write_config(tmp_path / "bootstrap.json", _typed_bootstrap())
        monkeypatch.setenv(CLIENT_API_TYPE_KEY, ClientAPIType.EX_PROCESS_API.value)

        with pytest.raises(ValueError, match="declares 'CELL_API'.*CLIENT_API_TYPE.*'EX_PROCESS_API'"):
            APIContext(config_file=config_file)

    def test_context_shutdown_is_idempotent(self, tmp_path, monkeypatch):
        config_file = _write_config(tmp_path / "bootstrap.json", _typed_bootstrap())
        monkeypatch.delenv(CLIENT_API_TYPE_KEY, raising=False)

        with patch("nvflare.client.cell.api.CellClientAPI") as cell_api_cls:
            ctx = APIContext(config_file=config_file)
            ctx.shutdown()
            ctx.shutdown()

        assert ctx.is_shutdown
        cell_api_cls.return_value.shutdown.assert_called_once_with()
        assert flare_api._runtime_shutdown is True


class _StubContext:
    def __init__(self, rank=None, config_file=None, api_type=ClientAPIType.IN_PROCESS_API):
        self.rank = rank
        self.config_file = config_file
        self.api_type = api_type
        self.is_shutdown = False
        self.shutdown_calls = 0

    def shutdown(self):
        if self.is_shutdown:
            return
        self.shutdown_calls += 1
        self.is_shutdown = True


class TestGlobalContextLifecycle:
    def test_backend_closed_in_process_context_is_recreated_for_later_job(self, monkeypatch):
        monkeypatch.delenv(CLIENT_API_TYPE_KEY, raising=False)
        first_api = MagicMock(spec=InProcessClientAPI)
        first_api.closed = False
        second_api = MagicMock(spec=InProcessClientAPI)
        second_api.closed = False

        with patch("nvflare.client.api_context.data_bus.get_data", side_effect=[first_api, second_api]):
            first = flare_api.init(rank=0)
            # InProcessBackend.finalize() owns this close; it cannot call APIContext.shutdown().
            first_api.closed = True
            second = flare_api.init(rank=0)

        assert second is not first
        assert second.api is second_api
        assert flare_api.default_context is second

    def test_cell_shutdown_rejects_same_process_reinitialization(self, monkeypatch, tmp_path):
        created = []

        def create_context(rank=None, config_file=None):
            ctx = _StubContext(rank=rank, config_file=config_file, api_type=ClientAPIType.CELL_API)
            created.append(ctx)
            return ctx

        monkeypatch.setattr(flare_api, "APIContext", create_context)
        config_file = str(tmp_path / "attach-bootstrap.json")

        first = flare_api.init(rank=0, config_file=config_file)
        flare_api.shutdown(first)

        assert first.shutdown_calls == 1
        with pytest.raises(RuntimeError, match="cannot be reinitialized.*same process"):
            flare_api.init(rank=0, config_file=config_file)
        assert created == [first]

    def test_stopped_thread_binding_cannot_resolve_or_stop_successor_context(self, monkeypatch):
        monkeypatch.setattr(flare_api, "APIContext", _StubContext)
        old_ready = threading.Event()
        old_stopped = threading.Event()
        successor_ready = threading.Event()
        old_state = {}

        def old_trainer():
            old_state["context"] = flare_api.init(rank=0)
            old_ready.set()
            flare_api.shutdown()
            old_stopped.set()
            assert successor_ready.wait(2.0)
            try:
                flare_api.get_context()
            except Exception as e:
                old_state["lookup_error"] = e
            # This repeated module-level shutdown must remain bound to the old tombstone.
            flare_api.shutdown()

        trainer = threading.Thread(target=old_trainer)
        trainer.start()
        assert old_ready.wait(2.0)
        assert old_stopped.wait(2.0)

        successor = flare_api.init(rank=0)
        successor_ready.set()
        trainer.join(timeout=2.0)

        assert not trainer.is_alive()
        assert isinstance(old_state.get("lookup_error"), RuntimeError)
        assert "Thread-bound APIContext has been shut down" in str(old_state["lookup_error"])
        assert successor.shutdown_calls == 0
        assert flare_api.get_context() is successor

    def test_unbound_helper_thread_can_use_global_default_context(self, monkeypatch):
        monkeypatch.setattr(flare_api, "APIContext", _StubContext)
        context = flare_api.init()
        observed = []

        helper = threading.Thread(target=lambda: observed.append(flare_api.get_context()))
        helper.start()
        helper.join(timeout=2.0)

        assert not helper.is_alive()
        assert observed == [context]

    @pytest.mark.parametrize("api_type", [ClientAPIType.IN_PROCESS_API, ClientAPIType.EX_PROCESS_API])
    def test_non_cell_shutdown_evicts_and_reinitializes_context(self, monkeypatch, api_type):
        created = []

        def create_context(rank=None, config_file=None):
            ctx = _StubContext(rank=rank, config_file=config_file, api_type=api_type)
            created.append(ctx)
            return ctx

        monkeypatch.setattr(flare_api, "APIContext", create_context)
        first = flare_api.init(rank=0)

        flare_api.shutdown(first)
        second = flare_api.init(rank=0)

        assert first.shutdown_calls == 1
        assert first not in flare_api.context_dict.values()
        assert second is not first
        assert created == [first, second]
        assert flare_api._runtime_shutdown is False

    def test_shutdown_default_removes_it_from_default_lookup(self, monkeypatch):
        monkeypatch.setattr(flare_api, "APIContext", _StubContext)
        ctx = flare_api.init()

        flare_api.shutdown()

        assert ctx.is_shutdown
        with pytest.raises(RuntimeError, match="Thread-bound APIContext has been shut down"):
            flare_api.get_context()

    def test_repeated_explicit_shutdown_is_idempotent_and_context_stays_evicted(self, monkeypatch):
        monkeypatch.setattr(flare_api, "APIContext", _StubContext)
        ctx = flare_api.init()

        flare_api.shutdown(ctx)
        flare_api.shutdown(ctx)

        assert ctx.shutdown_calls == 1
        assert ctx not in flare_api.context_dict.values()
        with pytest.raises(RuntimeError, match="APIContext has been shut down"):
            flare_api.get_context(ctx)

    def test_repeated_default_shutdown_is_idempotent(self, monkeypatch):
        monkeypatch.setattr(flare_api, "APIContext", _StubContext)
        ctx = flare_api.init()

        flare_api.shutdown()
        flare_api.shutdown()

        assert ctx.shutdown_calls == 1
        assert flare_api.default_context is None

    def test_default_shutdown_leaves_an_older_cached_context_usable(self, monkeypatch):
        monkeypatch.setattr(flare_api, "APIContext", _StubContext)
        older = flare_api.init(rank=0)
        current = flare_api.init(rank=1)

        flare_api.shutdown()
        flare_api.shutdown()

        assert current.shutdown_calls == 1
        assert older.shutdown_calls == 0
        assert flare_api.default_context is None
        assert flare_api.get_context(older) is older

    def test_stopped_cached_non_cell_context_is_replaced_without_public_shutdown(self, monkeypatch):
        monkeypatch.setattr(flare_api, "APIContext", _StubContext)
        first = flare_api.init()
        first.is_shutdown = True

        second = flare_api.init()

        assert second is not first
        assert flare_api.default_context is second

    def test_stopped_cached_cell_context_retires_runtime_for_every_key(self, monkeypatch):
        def create_context(rank=None, config_file=None):
            return _StubContext(rank=rank, config_file=config_file, api_type=ClientAPIType.CELL_API)

        monkeypatch.setattr(flare_api, "APIContext", create_context)
        first = flare_api.init()
        first.is_shutdown = True

        with pytest.raises(RuntimeError, match="cannot be reinitialized after shutdown"):
            flare_api.init(rank=1, config_file="different-bootstrap.json")
        assert flare_api._runtime_shutdown is True

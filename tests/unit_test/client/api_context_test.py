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
from unittest.mock import MagicMock, patch

import pytest

from nvflare.client.api_context import APIContext, ClientAPIType
from nvflare.client.api_spec import CLIENT_API_TYPE_KEY
from nvflare.client.cell.bootstrap import (
    BOOTSTRAP_FILE_ENV_VAR,
    BOOTSTRAP_SCHEMA_VERSION,
    EXTERNAL_PROCESS_EXECUTION_MODE,
    BootstrapKey,
)
from nvflare.client.in_process.api import InProcessClientAPI


def _write_config(path, config):
    path.write_text(json.dumps(config))
    return str(path)


def _typed_bootstrap():
    return {
        BootstrapKey.SCHEMA_VERSION: BOOTSTRAP_SCHEMA_VERSION,
        BootstrapKey.EXECUTION_MODE: EXTERNAL_PROCESS_EXECUTION_MODE,
    }


class TestAPIContextSelection:
    def test_explicit_typed_bootstrap_selects_cell_and_passes_exact_path(self, tmp_path, monkeypatch):
        config_file = _write_config(tmp_path / "attach-bootstrap.json", _typed_bootstrap())
        monkeypatch.delenv(CLIENT_API_TYPE_KEY, raising=False)

        with patch("nvflare.client.cell.api.CellClientAPI") as cell_api_cls:
            ctx = APIContext(rank="0", config_file=config_file)

        cell_api_cls.assert_called_once_with(bootstrap_file=config_file)
        cell_api_cls.return_value.init.assert_called_once_with(rank="0")
        assert ctx.api is cell_api_cls.return_value

    def test_bootstrap_env_selects_cell_without_duplicate_api_type(self, tmp_path, monkeypatch):
        config_file = _write_config(tmp_path / "launch-bootstrap.json", _typed_bootstrap())
        monkeypatch.setenv(BOOTSTRAP_FILE_ENV_VAR, config_file)
        monkeypatch.delenv(CLIENT_API_TYPE_KEY, raising=False)

        with patch("nvflare.client.cell.api.CellClientAPI") as cell_api_cls:
            APIContext(rank="0")

        cell_api_cls.assert_called_once_with(bootstrap_file=config_file)

    def test_bootstrap_env_overrides_untyped_explicit_legacy_config(self, tmp_path, monkeypatch):
        legacy_file = _write_config(tmp_path / "client_api_config.json", {"TASK_EXCHANGE": {}})
        bootstrap_file = _write_config(tmp_path / "launch-bootstrap.json", _typed_bootstrap())
        monkeypatch.setenv(BOOTSTRAP_FILE_ENV_VAR, bootstrap_file)
        monkeypatch.delenv(CLIENT_API_TYPE_KEY, raising=False)

        with patch("nvflare.client.cell.api.CellClientAPI") as cell_api_cls:
            APIContext(rank="0", config_file=legacy_file)

        cell_api_cls.assert_called_once_with(bootstrap_file=bootstrap_file)

    def test_env_cell_selection_without_explicit_config_keeps_bootstrap_env_behavior(self, monkeypatch):
        monkeypatch.setenv(CLIENT_API_TYPE_KEY, ClientAPIType.CELL_API.value)

        with patch("nvflare.client.cell.api.CellClientAPI") as cell_api_cls:
            APIContext(rank="0")

        cell_api_cls.assert_called_once_with()

    def test_legacy_ex_process_config_still_uses_env_selection(self, tmp_path, monkeypatch):
        config_file = _write_config(tmp_path / "legacy.json", {"TASK_EXCHANGE": {}})
        monkeypatch.setenv(CLIENT_API_TYPE_KEY, ClientAPIType.EX_PROCESS_API.value)

        with patch("nvflare.client.api_context.ExProcessClientAPI") as ex_process_api_cls:
            APIContext(rank="0", config_file=config_file)

        ex_process_api_cls.assert_called_once_with(config_file=config_file)
        ex_process_api_cls.return_value.init.assert_called_once_with(rank="0")

    def test_untyped_explicit_config_preserves_default_in_process_selection(self, tmp_path, monkeypatch):
        config_file = _write_config(tmp_path / "legacy.json", {"TASK_EXCHANGE": {}})
        monkeypatch.delenv(CLIENT_API_TYPE_KEY, raising=False)
        in_process_api = MagicMock(spec=InProcessClientAPI)

        with patch("nvflare.client.api_context.data_bus.get_data", return_value=in_process_api):
            ctx = APIContext(rank="0", config_file=config_file)

        assert ctx.api is in_process_api
        in_process_api.init.assert_called_once_with(rank="0")

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

    def test_context_manager_shuts_down_cell_api(self, tmp_path, monkeypatch):
        config_file = _write_config(tmp_path / "bootstrap.json", _typed_bootstrap())
        monkeypatch.delenv(CLIENT_API_TYPE_KEY, raising=False)

        with patch("nvflare.client.cell.api.CellClientAPI") as cell_api_cls:
            with APIContext(config_file=config_file):
                pass

        cell_api_cls.return_value.shutdown.assert_called_once_with()

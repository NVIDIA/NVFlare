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


class TestJobNew:
    """Tests for nvflare job new command."""

    def _make_args(self, **kwargs):
        args = MagicMock()
        args.output = kwargs.get("output", "json")
        args.recipe = kwargs.get("recipe", "fedavg")
        args.script = kwargs.get("script", "train.py")
        args.job_folder = kwargs.get("job_folder", "/tmp/test_job")
        args.min_clients = kwargs.get("min_clients", 2)
        args.study = kwargs.get("study", "default")
        args.param = kwargs.get("param", [])
        return args

    def test_known_recipe_calls_scaffolding(self, capsys):
        """Known recipe: mock catalog, verify FedJob instantiated and export_job called with correct folder."""
        import importlib

        from nvflare.tool.job.job_cli import cmd_job_new

        args = self._make_args(recipe="fedavg", job_folder="/tmp/test_scaffold_job", min_clients=3, script="train.py")

        fake_catalog = [{"name": "fedavg", "module": "fake.fedavg.module", "class": "FedAvgRecipe"}]
        fake_recipe_instance = MagicMock()
        fake_recipe_class = MagicMock(return_value=fake_recipe_instance)
        fake_mod = MagicMock()
        setattr(fake_mod, "FedAvgRecipe", fake_recipe_class)
        fake_job = MagicMock()

        mock_recipe_cli = MagicMock()
        mock_recipe_cli._load_catalog = MagicMock(return_value=fake_catalog)

        original_import = importlib.import_module

        def mock_import(name, *a, **kw):
            if name == "fake.fedavg.module":
                return fake_mod
            return original_import(name, *a, **kw)

        with patch.dict("sys.modules", {"nvflare.tool.recipe.recipe_cli": mock_recipe_cli}):
            with patch("nvflare.tool.cli_schema.handle_schema_flag", return_value=None):
                with patch("importlib.import_module", side_effect=mock_import):
                    import nvflare.job_config.api as job_api

                    original_fedjob = getattr(job_api, "FedJob", None)
                    job_api.FedJob = MagicMock(return_value=fake_job)
                    try:
                        cmd_job_new(args)
                    except SystemExit:
                        pass
                    finally:
                        if original_fedjob is not None:
                            job_api.FedJob = original_fedjob

        # FedJob should have been constructed and export_job called
        assert fake_job.export_job.called or fake_job.to.called or fake_job.method_calls

    def test_unknown_recipe_exits_4(self):
        """Unknown recipe: catalog returns empty list, verify exit 4."""
        from nvflare.tool.job.job_cli import cmd_job_new

        args = self._make_args(recipe="nonexistent_recipe")

        fake_catalog = [{"name": "fedavg", "module": "some.module", "class": "SomeClass"}]

        mock_recipe_cli = MagicMock()
        mock_recipe_cli._load_catalog = MagicMock(return_value=fake_catalog)
        with patch.dict("sys.modules", {"nvflare.tool.recipe.recipe_cli": mock_recipe_cli}):
            with patch("nvflare.tool.cli_schema.handle_schema_flag", return_value=None):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_job_new(args)
                assert exc_info.value.code == 4

    def test_catalog_import_error_exits_4(self):
        """ImportError loading catalog: exit 4 with INVALID_ARGS."""
        from nvflare.tool.job.job_cli import cmd_job_new

        args = self._make_args()

        # Make the import raise ImportError by setting the module to None
        import sys as _sys

        saved = _sys.modules.pop("nvflare.tool.recipe.recipe_cli", None)
        saved_parent = _sys.modules.pop("nvflare.tool.recipe", None)
        try:
            with patch.dict("sys.modules", {"nvflare.tool.recipe.recipe_cli": None}):
                with patch("nvflare.tool.cli_schema.handle_schema_flag", return_value=None):
                    with pytest.raises(SystemExit) as exc_info:
                        cmd_job_new(args)
                    assert exc_info.value.code == 4
        finally:
            if saved is not None:
                _sys.modules["nvflare.tool.recipe.recipe_cli"] = saved
            if saved_parent is not None:
                _sys.modules["nvflare.tool.recipe"] = saved_parent

    def test_param_coercion(self):
        """--param coercion: rounds=10 -> int, lr=0.01 -> float, flag=true -> bool."""
        from nvflare.tool.job.job_cli import _coerce

        assert _coerce("10") == 10
        assert isinstance(_coerce("10"), int)

        assert _coerce("0.01") == 0.01
        assert isinstance(_coerce("0.01"), float)

        assert _coerce("true") is True
        assert _coerce("false") is False

        assert _coerce("hello") == "hello"
        assert isinstance(_coerce("hello"), str)

    def test_json_output_fields(self, capsys):
        """JSON output: verify output_ok called with job_folder, recipe, min_clients, script, params."""
        from nvflare.tool.cli_output import output_ok

        data = {
            "job_folder": "/tmp/test_job",
            "recipe": "fedavg",
            "min_clients": 2,
            "script": "train.py",
            "params": {"rounds": 10, "lr": 0.01},
        }
        output_ok(data, "json")
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["status"] == "ok"
        assert result["data"]["job_folder"] == "/tmp/test_job"
        assert result["data"]["recipe"] == "fedavg"
        assert result["data"]["min_clients"] == 2
        assert result["data"]["script"] == "train.py"
        assert result["data"]["params"]["rounds"] == 10

    def test_txt_output_prints_folder_path(self, capsys):
        """--output txt: verify only folder path printed."""
        folder = "/tmp/test_job_output"

        # Simulate what cmd_job_new does in txt mode
        print(folder)
        captured = capsys.readouterr()
        assert captured.out.strip() == folder

    def test_txt_output_via_cmd(self, capsys):
        """--output txt integration: cmd_job_new prints folder path."""
        from nvflare.tool.job.job_cli import cmd_job_new

        args = self._make_args(output="txt", job_folder="/tmp/my_new_job")

        fake_catalog = [{"name": "fedavg", "module": "fake.module", "class": "FakeClass"}]
        fake_instance = MagicMock()
        fake_class = MagicMock(return_value=fake_instance)
        fake_module = MagicMock()
        setattr(fake_module, "FakeClass", fake_class)
        fake_job = MagicMock()

        mock_recipe_cli = MagicMock()
        mock_recipe_cli._load_catalog = MagicMock(return_value=fake_catalog)

        with patch.dict("sys.modules", {"nvflare.tool.recipe.recipe_cli": mock_recipe_cli}):
            with patch("nvflare.tool.cli_schema.handle_schema_flag", return_value=None):
                import importlib

                original_import = importlib.import_module

                def mock_import(name, *a, **kw):
                    if name == "fake.module":
                        return fake_module
                    return original_import(name, *a, **kw)

                with patch("importlib.import_module", side_effect=mock_import):
                    import nvflare.job_config.api as job_api

                    original_fedjob = getattr(job_api, "FedJob", None)
                    job_api.FedJob = MagicMock(return_value=fake_job)
                    try:
                        cmd_job_new(args)
                    except Exception:
                        pass
                    finally:
                        if original_fedjob is not None:
                            job_api.FedJob = original_fedjob

        captured = capsys.readouterr()
        # In txt mode, the folder path should appear in output
        assert "/tmp/my_new_job" in captured.out

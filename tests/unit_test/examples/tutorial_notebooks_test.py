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

import ast
import json
import re
import shlex
import sys
import textwrap
from pathlib import Path
from unittest.mock import Mock, call
from urllib.parse import urlsplit

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TUTORIALS = _REPO_ROOT / "examples/tutorials"
_HELLO_PT = _REPO_ROOT / "examples/hello-world/hello-pt"
_NOTEBOOKS = ("flare_simulator", "job_recipe", "setup_poc", "nvflare_cli", "flare_api", "logging")


def _cells(name):
    return json.loads((_TUTORIALS / f"{name}.ipynb").read_text(encoding="utf-8"))["cells"]


def _source(cell):
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def _cell(name, cell_id):
    return _source(next(cell for cell in _cells(name) if cell["id"] == cell_id))


@pytest.mark.parametrize("name", _NOTEBOOKS)
def test_notebook_documentation_pages_exist(name):
    for cell in _cells(name):
        for url in re.findall(r"https://nvflare\.readthedocs\.io/en/(?:main|latest)/[^\s)>]+", _source(cell)):
            path = urlsplit(url).path.split("/", 3)[-1]
            if path.endswith(".html") and not path.startswith("apidocs/"):
                assert (_REPO_ROOT / "docs" / f"{path[:-5]}.rst").is_file(), url


def test_recipe_notebook_uses_maintained_hello_pt(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    monkeypatch.chdir(_TUTORIALS)
    monkeypatch.setattr(sys, "path", list(sys.path))
    # Record the previous entry even when absent, so the imported model is
    # removed at teardown instead of leaking into unrelated example tests.
    monkeypatch.setitem(sys.modules, "model", None)
    monkeypatch.delitem(sys.modules, "model", raising=False)
    scope = {}
    exec(_cell("job_recipe", "basic-example-code"), scope)
    recipe = scope["recipe"]
    assert Path(sys.modules["model"].__file__).resolve() == _HELLO_PT / "model.py"
    assert Path(recipe.train_script) == _HELLO_PT / "client.py"
    assert "--dataset synthetic" in recipe.train_args

    recipe.export(job_dir=str(tmp_path))
    custom = tmp_path / "hello-pt/app/custom"
    for filename in ("client.py", "model.py", "prepare_data.py"):
        assert (custom / filename).read_bytes() == (_HELLO_PT / filename).read_bytes()

    # A subprocess starts with its current directory on the import path. Keeping
    # it at examples/tutorials would shadow the exported Hello PyTorch model.
    with scope["hello_pt_working_directory"]():
        assert Path.cwd() == _HELLO_PT
    assert Path.cwd() == _TUTORIALS
    with pytest.raises(RuntimeError), scope["hello_pt_working_directory"]():
        raise RuntimeError("deployment failed")
    assert Path.cwd() == _TUTORIALS
    for cell in _cells("job_recipe"):
        source = _source(cell)
        if cell["cell_type"] == "code" and "recipe.execute(" in source:
            tree = ast.parse(source)
            assert any(
                isinstance(node, ast.With) and "recipe.execute(" in ast.unparse(node) for node in ast.walk(tree)
            ), cell["id"]

    options = _cell("job_recipe", "b7b124e7")
    model_args = json.loads(re.search(r'"args": (\{[^}]*\})', options).group(1))
    scope["SimpleNetwork"](**model_args)


def test_recipe_production_kit_matches_provision_workspace():
    provision = _cell("job_recipe", "d3451070-14ff-4fdd-bd79-e87413f5d30c")
    argv = shlex.split(provision.removeprefix("!"))
    workspace = Path(argv[argv.index("-w") + 1])
    tree = ast.parse(_cell("job_recipe", "67c99d67"))
    admin_kit = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ADMIN_KIT" for t in node.targets)
    )
    assert Path(admin_kit).is_relative_to(workspace)


def test_api_notebook_resubmits_and_cleans_up_both_jobs(monkeypatch):
    from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
    from nvflare.fuel.flare_api import flare_api

    monkeypatch.chdir(_TUTORIALS)
    session = Mock()
    session.submit_job.side_effect = ["first-job", "repeat-job"]
    session.list_jobs.return_value = []
    session.abort_job.return_value = "Abort signal sent"
    export = Mock()
    monkeypatch.setattr(NumpyFedAvgRecipe, "export", export)
    monkeypatch.setattr(flare_api, "new_secure_session", Mock(return_value=session))
    scope = {}
    for cell in _cells("flare_api"):
        if cell["cell_type"] == "code":
            exec(_source(cell), scope)

    job_dir = "/tmp/nvflare/flare_api_tutorial"
    export.assert_called_once_with(job_dir)
    assert scope["recipe"].name == "hello-numpy"
    assert (_REPO_ROOT / "examples/hello-world/hello-numpy/client.py").is_file()
    assert session.submit_job.call_count == 2
    assert all(call.args == (f"{job_dir}/hello-numpy",) for call in session.submit_job.call_args_list)
    session.clone_job.assert_not_called()
    # The tutorial calls abort_job() once; any startup retry is handled inside the FLARE API.
    assert session.abort_job.call_args_list == [call("repeat-job")]
    assert [call.args[0] for call in session.monitor_job.call_args_list] == ["first-job", "repeat-job"]
    assert [call.args[0] for call in session.delete_job.call_args_list] == ["repeat-job", "first-job"]
    session.close.assert_called_once()


def test_logging_custom_logger_example_executes():
    source = "\n".join(_source(cell) for cell in _cells("logging"))
    block = next(block for block in re.findall(r"```python\n(.*?)```", source, re.DOTALL) if "custom_logger" in block)
    scope = {"__name__": "tutorial_logging_example"}
    exec(textwrap.dedent(block), scope)
    assert scope["logger"].name.startswith("custom.")
    assert scope["MyClass"]().logger.name.startswith("custom.")

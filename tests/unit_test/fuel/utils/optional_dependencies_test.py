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

import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nvflare.fuel.utils import optional_dependencies
from nvflare.fuel.utils.optional_dependencies import (
    DASHBOARD_OPTIONAL_MODULES,
    MissingOptionalDependencyError,
    optional_dependency_import,
    require_optional_dependencies,
)


@pytest.fixture
def mock_distributions(monkeypatch):
    def configure(versions, module_path=optional_dependencies.__file__):
        def lookup(name):
            if name not in versions:
                raise PackageNotFoundError(name)
            return SimpleNamespace(version=versions[name], locate_file=lambda _: module_path)

        monkeypatch.setattr(optional_dependencies, "distribution", lookup)

    return configure


def _blocked_web_modules_code() -> str:
    return dedent(
        """
        import importlib.abc
        import sys

        class _BlockWebModules(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                blocked = ("flask", "flask_jwt_extended", "flask_sqlalchemy", "werkzeug")
                if any(name == module or name.startswith(f"{module}.") for module in blocked):
                    raise ModuleNotFoundError(f"No module named '{name}'", name=name)
                return None

        sys.meta_path.insert(0, _BlockWebModules())
        """
    )


def test_require_optional_dependencies_lists_missing_modules(mock_distributions):
    mock_distributions({"nvflare": "2.10.0"})
    with (
        patch("nvflare.fuel.utils.optional_dependencies.find_spec", return_value=None),
        pytest.raises(MissingOptionalDependencyError, match=r"nvflare\[dashboard\]") as exc_info,
    ):
        require_optional_dependencies("NVFlare Dashboard", "dashboard", DASHBOARD_OPTIONAL_MODULES)

    assert "flask, flask_jwt_extended, flask_sqlalchemy" in str(exc_info.value)


@pytest.mark.parametrize("extra", ("dashboard", "edge-http"))
@pytest.mark.parametrize(
    ("name", "version"),
    (
        ("nvflare", "2.10.1"),
        ("nvflare-nightly", "2.10.0.dev20261029"),
        ("nvflare-nightly", "2.10.0.dev0+125.gd05110ab4.dirty"),
    ),
)
def test_recovery_guidance_preserves_distribution_and_version(mock_distributions, name, version, extra):
    mock_distributions({name: version})
    expected = f"python -m pip install '{name}[{extra}]=={version}'"
    with patch.object(optional_dependencies, "find_spec", return_value=None):
        with pytest.raises(MissingOptionalDependencyError) as preflight_error:
            require_optional_dependencies("feature", extra, ("flask",))
    with pytest.raises(MissingOptionalDependencyError) as import_error:
        with optional_dependency_import("feature", extra, ("flask",)):
            raise ModuleNotFoundError("No module named 'flask'", name="flask")

    assert expected in str(preflight_error.value)
    assert expected in str(import_error.value)


@pytest.mark.parametrize(
    "versions",
    ({}, {"nvflare": "2.10.0", "nvflare-nightly": "2.10.0.dev20261029"}, {"nvflare": ""}),
)
def test_recovery_guidance_does_not_guess_distribution(mock_distributions, versions):
    mock_distributions(versions)
    with patch.object(optional_dependencies, "find_spec", return_value=None):
        with pytest.raises(MissingOptionalDependencyError) as error:
            require_optional_dependencies("feature", "dashboard", ("flask",))

    assert "same NVFlare wheel or source checkout with the 'dashboard' extra" in str(error.value)
    assert "pip install" not in str(error.value)


def test_recovery_guidance_does_not_target_a_different_checkout(mock_distributions, tmp_path):
    mock_distributions({"nvflare": "2.10.0"}, tmp_path / "other-checkout" / "optional_dependencies.py")
    with patch.object(optional_dependencies, "find_spec", return_value=None):
        with pytest.raises(MissingOptionalDependencyError) as error:
            require_optional_dependencies("feature", "edge-http", ("flask",))

    assert "same NVFlare wheel or source checkout with the 'edge-http' extra" in str(error.value)
    assert "pip install" not in str(error.value)


def test_optional_dependency_import_does_not_mask_unrelated_import_errors():
    with pytest.raises(ModuleNotFoundError, match="unrelated_dependency") as exc_info:
        with optional_dependency_import("NVFlare Dashboard", "dashboard", DASHBOARD_OPTIONAL_MODULES):
            raise ModuleNotFoundError("No module named 'unrelated_dependency'", name="unrelated_dependency")

    assert "nvflare[dashboard]" not in str(exc_info.value)


def test_core_runtime_imports_do_not_require_web_dependencies():
    code = _blocked_web_modules_code() + dedent(
        """
        import nvflare
        import nvflare.cli
        import nvflare.dashboard
        import nvflare.dashboard.cli
        import nvflare.edge.web
        import nvflare.edge.web.service.client
        import nvflare.edge.web.service.server
        import nvflare.edge.widgets.api_service
        import nvflare.private.fed.app.client.client_train
        import nvflare.private.fed.app.server.server_train
        """
    )

    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    ("module", "extra"),
    [
        ("nvflare.dashboard.application", "dashboard"),
        ("nvflare.edge.web.routing_proxy", "edge-http"),
        ("nvflare.edge.web.web_server", "edge-http"),
    ],
)
def test_web_runtime_entries_report_their_install_extra(module, extra):
    code = _blocked_web_modules_code() + dedent(
        f"""
        import importlib

        try:
            importlib.import_module({module!r})
        except ImportError as error:
            assert "[{extra}]" in str(error) or "'{extra}' extra" in str(error), str(error)
        else:
            raise AssertionError("web runtime import unexpectedly succeeded")
        """
    )

    subprocess.run([sys.executable, "-c", code], check=True)

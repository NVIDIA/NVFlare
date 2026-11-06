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
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
from pathlib import Path
from typing import Iterator, Sequence

from packaging.version import InvalidVersion, Version

DASHBOARD_OPTIONAL_MODULES = ("flask", "flask_jwt_extended", "flask_sqlalchemy")
EDGE_HTTP_OPTIONAL_MODULES = ("flask",)


class MissingOptionalDependencyError(ImportError):
    """Raised when an explicitly requested feature is not installed."""


def _installation_hint(extra: str) -> str:
    fallback = f"Reinstall the same NVFlare wheel or source checkout with the '{extra}' extra."
    installations = []
    for name in ("nvflare", "nvflare-nightly"):
        try:
            installed = distribution(name)
        except PackageNotFoundError:
            continue
        module_path = installed.locate_file("nvflare/fuel/utils/optional_dependencies.py")
        if Path(module_path).resolve() == Path(__file__).resolve() and installed.version:
            installations.append((name, installed))
    if len(installations) == 1:
        name, installed = installations[0]
        try:
            version = Version(installed.version)
            direct_url = installed.read_text("direct_url.json")
            source = json.loads(direct_url) if direct_url else None
        except (InvalidVersion, json.JSONDecodeError):
            return fallback
        # Container builds install non-editably from a temporary source mount.
        directory = source.get("dir_info") if isinstance(source, dict) else None
        built_from_source = isinstance(directory, dict) and directory.get("editable", False) is False
        # --find-links and private indexes need not write direct_url.json.
        # Their development/local versions must not be assumed to exist on PyPI.
        if (
            name == "nvflare"
            and not version.is_devrelease
            and not version.local
            and (not direct_url or built_from_source)
        ):
            return f"Install them with: python -m pip install 'nvflare[{extra}]=={installed.version}'"
    return fallback


def _missing_dependency_error(feature: str, extra: str, missing_modules: Sequence[str]):
    missing = ", ".join(missing_modules)
    return MissingOptionalDependencyError(
        f"{feature} requires optional dependencies ({missing}). {_installation_hint(extra)} "
        f"For a container image, include the '{extra}' extra at image build time."
    )


def require_optional_dependencies(feature: str, extra: str, required_modules: Sequence[str]) -> None:
    """Fail with installation guidance when declared optional modules are absent."""
    missing_modules = tuple(module for module in required_modules if find_spec(module) is None)
    if missing_modules:
        raise _missing_dependency_error(feature, extra, missing_modules)


@contextmanager
def optional_dependency_import(feature: str, extra: str, required_modules: Sequence[str]) -> Iterator[None]:
    """Translate only missing declared modules into an actionable error."""
    try:
        yield
    except ModuleNotFoundError as error:
        if error.name not in required_modules:
            raise
        raise _missing_dependency_error(feature, extra, (error.name,)) from error

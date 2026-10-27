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

from contextlib import contextmanager
from importlib.util import find_spec
from typing import Iterator, Sequence

DASHBOARD_OPTIONAL_MODULES = ("flask", "flask_jwt_extended", "flask_sqlalchemy")
EDGE_HTTP_OPTIONAL_MODULES = ("flask",)


class MissingOptionalDependencyError(ImportError):
    """Raised when an explicitly requested feature is not installed."""


def _missing_dependency_error(feature: str, extra: str, missing_modules: Sequence[str]):
    missing = ", ".join(missing_modules)
    return MissingOptionalDependencyError(
        f"{feature} requires optional dependencies ({missing}). "
        f"Install them with: python -m pip install 'nvflare[{extra}]'"
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

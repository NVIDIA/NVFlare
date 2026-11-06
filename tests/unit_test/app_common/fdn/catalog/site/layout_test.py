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
from pathlib import Path

import nvflare.app_common.fdn.catalog as catalog_pkg

_ROOT = Path(catalog_pkg.__file__).parent


def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


def test_site_code_does_not_import_server_or_admin_code():
    for path in (_ROOT / "site").glob("*.py"):
        bad = [
            m
            for m in _imports(path)
            if m.startswith(("nvflare.app_common.fdn.catalog.server", "nvflare.app_common.fdn.catalog.admin"))
        ]
        assert not bad, f"{path.name} imports {bad}"


def test_common_code_imports_only_common_code():
    for path in (_ROOT / "common").glob("*.py"):
        bad = [m for m in _imports(path) if m.startswith("nvflare.app_common.fdn.catalog.") and ".common" not in m]
        assert not bad, f"{path.name} imports {bad}"

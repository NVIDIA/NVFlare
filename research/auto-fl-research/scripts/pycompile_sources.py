#!/usr/bin/env python3
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

"""Compile repository Python sources without shell argv expansion."""

import argparse
import os
import py_compile
import sys
from pathlib import Path

EXCLUDED_DIRS = {
    ".devcontainer",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "env",
    "results",
    "site-packages",
    "venv",
    "worktrees",
}


def iter_python_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in EXCLUDED_DIRS and not name.endswith(".egg-info")]
        for filename in filenames:
            if filename.endswith(".py"):
                yield Path(dirpath) / filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"source root not found: {root}")

    if not sys.pycache_prefix:
        sys.pycache_prefix = os.environ.get("PYTHONPYCACHEPREFIX", "/tmp/auto-fl-pycache")

    failures = []
    count = 0
    for path in iter_python_files(root):
        count += 1
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append((path, exc))

    if failures:
        print("ERROR: Python syntax validation failed:")
        for path, exc in failures:
            print(f"  - {path}: {exc.msg}")
        return 1

    print(f"OK: compiled {count} Python source files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

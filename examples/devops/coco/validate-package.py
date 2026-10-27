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

"""Developer checks for a Git source checkout; never executes deployment stages or uses the network."""

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-tree", action="store_true", help="validate the current source checkout (default)")
    parser.parse_args()
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11+ is required")
    # Include newly added source and omit pending deletions without modifying
    # the index. Ignored deployment inputs and credentials stay outside checks.
    try:
        entries = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "."],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).split("\0")
    except OSError:
        raise SystemExit("Git is required for package validation; install Git and run from an NVFlare source checkout.")
    except subprocess.CalledProcessError:
        raise SystemExit("Package validation requires an NVFlare Git source checkout; archive copies are unsupported.")
    names = sorted({name for name in entries if name and ((ROOT / name).exists() or (ROOT / name).is_symlink())})
    if not {"validate-package.py", "README.md"} <= set(names):
        raise SystemExit("Package sources are excluded from Git; run validation from an NVFlare Git source checkout.")
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name:
            raise SystemExit(f"Unsafe source path: {name}")
        if any((ROOT / part).is_symlink() for part in (path, *path.parents)) or not (ROOT / path).is_file():
            raise SystemExit(f"Source must contain regular files: {name}")

    private = re.compile(
        r"\b[a-z]+[0-9]*-[0-9]{4}\.[a-z0-9]+\.[a-z0-9]+\.nvidia\.com|/localhome/[a-zA-Z0-9_-]+/|"
        r"[a-z0-9-]+\.[a-z0-9-]+\.cloudapp\.[a-z]+\.com|"
        r"-----BEGIN (?:RSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"
    )
    shell_count = python_count = embedded_count = 0
    for name in names:
        path = ROOT / name
        if path.suffix in (".pptx", ".pdf"):
            raise SystemExit(f"Generated slide exports are not part of this package: {name}")
        if path.name == "CURRENT-STATE.md":
            raise SystemExit("Keep package capabilities in the main README, not role state documents")
        content = path.read_text()
        if private.search(content):
            raise SystemExit(f"Potential private deployment identifier or key in {name}")
        if path.suffix == ".py":
            ast.parse(content, filename=name)
            python_count += 1
        if path.suffix == ".sh":
            subprocess.run(["bash", "-n", str(path)], check=True)
            if not os.access(path, os.X_OK):
                raise SystemExit(f"Shell executable mode is missing: {name}")
            shell_count += 1
            for match in re.finditer(r"<<['\"]?(PY[A-Z0-9_]*|CHECK)['\"]?[^\n]*\n(.*?)^\1\s*$", content, re.M | re.S):
                ast.parse(match[2], filename=f"{name}:embedded-python")
                embedded_count += 1
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", content):
                if "://" in target or target.startswith("#"):
                    continue
                target = target.split("#", 1)[0]
                if target and not (path.parent / target).exists():
                    raise SystemExit(f"Broken local document link in {name}: {target}")

    print(
        f"Validated {len(names)} public files, {shell_count} shell scripts, "
        f"{python_count} Python files and {embedded_count} embedded Python blocks."
    )
    print("Static checks only; no regression tests, remote deployment or hardware attestation were performed.")


if __name__ == "__main__":
    main()

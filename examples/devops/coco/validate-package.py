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

"""Offline publication checks; never executes deployment stages or uses the network."""

import ast
import hashlib
import os
import re
import subprocess
import sys
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    # Validation must not add bytecode caches containing local checkout paths.
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11+ is required")
    try:
        import yaml  # noqa: F401
    except ImportError:
        raise SystemExit("Install PyYAML in your chosen Python environment before validation") from None

    names = (ROOT / "PACKAGE-FILES.txt").read_text().splitlines()
    if len(names) != len(set(names)) or names != sorted(names):
        raise SystemExit("Public allowlist must be sorted and unique")
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name:
            raise SystemExit("Unsafe public allowlist entry")
    expected = set(names) | {"PACKAGE-SHA256SUMS"}
    found = set()
    for path in ROOT.rglob("*"):
        if path.is_symlink():
            raise SystemExit(f"Symlink is not publishable: {path.relative_to(ROOT)}")
        if path.is_file():
            found.add(path.relative_to(ROOT).as_posix())
    if expected != found:
        raise SystemExit(
            f"Public inventory mismatch: missing={sorted(expected - found)}, extra={sorted(found - expected)}"
        )

    checksums = []
    for line in (ROOT / "PACKAGE-SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or name not in names:
            raise SystemExit("Invalid checksum manifest")
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"Checksum mismatch: {name}")
        checksums.append(name)
    if checksums != names:
        raise SystemExit("Checksums must cover the exact sorted public allowlist")

    private = re.compile(
        r"\b[a-z]+[0-9]*-[0-9]{4}\.[a-z0-9]+\.[a-z0-9]+\.nvidia\.com|/localhome/[a-zA-Z0-9_-]+/|"
        r"[a-z0-9-]+\.[a-z0-9-]+\.cloudapp\.[a-z]+\.com|"
        r"-----BEGIN (?:RSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"
    )
    shell_count = python_count = embedded_count = 0
    for name in names:
        path = ROOT / name
        if path.suffix == ".pptx":
            with zipfile.ZipFile(path) as archive:
                content = "\n".join(archive.read(n).decode(errors="ignore") for n in archive.namelist())
        elif path.suffix == ".pdf":
            # PDF content was generated from the public slide sources. Compressed
            # streams still require a separate rendered/content review before publishing.
            content = path.read_bytes().decode(errors="ignore")
        else:
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

    result_ok = True
    for directory in ("admin/tests", "service/tests", "tests"):
        suite = unittest.TestLoader().discover(str(ROOT / directory))
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        result_ok = result.wasSuccessful() and result_ok
    if not result_ok:
        raise SystemExit("Offline tests failed")
    print(
        f"Validated {len(names)} public files, {shell_count} shell scripts, "
        f"{python_count} Python files and {embedded_count} embedded Python blocks."
    )
    print("No remote deployment or hardware attestation was performed.")


if __name__ == "__main__":
    main()

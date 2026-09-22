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

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUBLIC_DOCS = {
    "docs/coco-security-design-3-slides.md",
    "docs/coco-four-party-sequence.mmd",
    "docs/coco-four-party-sequence.html",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembled", action="store_true", help="validate generated role kits, not source wrappers")
    args = parser.parse_args()
    # Validation must not add bytecode caches containing local checkout paths.
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from role_kits import package_files, validate_layout

    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11+ is required")
    names = package_files(ROOT, assembled=args.assembled)
    validate_layout(ROOT, assembled=args.assembled)

    private = re.compile(
        r"\b[a-z]+[0-9]*-[0-9]{4}\.[a-z0-9]+\.[a-z0-9]+\.nvidia\.com|/localhome/[a-zA-Z0-9_-]+/|"
        r"[a-z0-9-]+\.[a-z0-9-]+\.cloudapp\.[a-z]+\.com|"
        r"-----BEGIN (?:RSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"
    )
    shell_count = python_count = embedded_count = 0
    for name in names:
        path = ROOT / name
        if name.startswith("docs/") and name not in PUBLIC_DOCS:
            raise SystemExit(f"Unapproved public document in docs/: {name}")
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

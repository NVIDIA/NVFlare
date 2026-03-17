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

"""Install requirements from a file, optionally excluding specific packages.

Usage:
    python install_requirements.py <requirements_file> [--exclude <pkg1,pkg2,...>] [--quiet]

Example:
    python install_requirements.py requirements.txt --exclude nvflare --quiet
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description="Install requirements, optionally excluding packages")
    parser.add_argument("requirements_file", help="Path to requirements.txt")
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated list of package name prefixes to exclude (case-insensitive)",
    )
    parser.add_argument("--quiet", action="store_true", help="Pass --quiet to pip")
    args = parser.parse_args()

    excluded = [e.strip().lower() for e in args.exclude.split(",") if e.strip()]

    with open(args.requirements_file) as f:
        lines = f.readlines()

    filtered = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if excluded:
            # Extract the package specifier before any inline comment for exclude matching
            spec = re.split(r"\s+#", stripped)[0].strip()
            if any(spec.lower().startswith(ex) for ex in excluded):
                continue
        filtered.append(line)

    if not filtered:
        print("No packages to install.")
        return

    req_dir = os.path.dirname(os.path.abspath(args.requirements_file))
    fd, tmp_path = tempfile.mkstemp(suffix=".txt", dir=req_dir)
    result = 1
    try:
        try:
            with os.fdopen(fd, "w") as tmp:
                tmp.writelines(filtered)
        except Exception:
            os.close(fd)
            raise
        cmd = [sys.executable, "-m", "pip", "install", "-r", tmp_path]
        if args.quiet:
            cmd.append("--quiet")
        result = subprocess.call(cmd)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass

    sys.exit(result)


if __name__ == "__main__":
    main()

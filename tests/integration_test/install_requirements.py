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
import subprocess
import sys


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

    packages = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pkg_lower = stripped.lower()
        if any(pkg_lower.startswith(ex) for ex in excluded):
            continue
        packages.append(stripped)

    if not packages:
        print("No packages to install.")
        return

    cmd = [sys.executable, "-m", "pip", "install"] + packages
    if args.quiet:
        cmd.append("--quiet")

    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

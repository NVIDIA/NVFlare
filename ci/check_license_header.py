#!/usr/bin/env python3
#
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

import argparse
import re
from pathlib import Path

DEFAULT_FOLDERS = ("nvflare", "examples", "tests", "integration", "research")
PUBLIC_DOMAIN_MARKER = "This file is released into the public domain."
EXCLUDED_FILE_NAMES = {"modeling_roberta.py"}

LICENSE_HEADER_PATTERN = re.compile(
    r"\A"
    r"(?:#!.*\n)?"
    r"(?:#.*coding[:=].*\n)?"
    r"(?:\n)*"
    r"# Copyright \(c\) \d{4}(?:-\d{4})?, NVIDIA CORPORATION\.  All rights reserved\.\n"
    r"#\n"
    r"# Licensed under the Apache License, Version 2\.0 \(the \"License\"\);\n"
    r"# you may not use this file except in compliance with the License\.\n"
    r"# You may obtain a copy of the License at\n"
    r"#\n"
    r"#     http://www\.apache\.org/licenses/LICENSE-2\.0\n"
    r"#\n"
    r"# Unless required by applicable law or agreed to in writing, software\n"
    r"# distributed under the License is distributed on an \"AS IS\" BASIS,\n"
    r"# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied\.\n"
    r"# See the License for the specific language governing permissions and\n"
    r"# limitations under the License\.\n"
)


def iter_python_files(folders: list[str]) -> list[Path]:
    files = []
    for folder in folders:
        root = Path(folder)
        if not root.exists():
            continue

        for file_path in root.rglob("*.py"):
            if any("protos" in part for part in file_path.parts):
                continue
            if file_path.name in EXCLUDED_FILE_NAMES:
                continue
            files.append(file_path)
    return sorted(files)


def has_valid_license_header(file_text: str) -> bool:
    if PUBLIC_DOMAIN_MARKER in file_text[:512]:
        return True
    return bool(LICENSE_HEADER_PATTERN.match(file_text))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate canonical license headers for Python files.")
    parser.add_argument("folders", nargs="*", default=list(DEFAULT_FOLDERS), help="Folders to recursively scan.")
    args = parser.parse_args()

    files_with_bad_headers = []
    for file_path in iter_python_files(args.folders):
        file_text = file_path.read_text(encoding="utf-8", errors="ignore")
        if not has_valid_license_header(file_text):
            files_with_bad_headers.append(file_path)

    if files_with_bad_headers:
        for file_path in files_with_bad_headers:
            print(file_path)
        print("License text not found or inconsistent on the above files.")
        print("Please fix them.")
        return 1

    print(f"All Python files in folder {' '.join(args.folders)} have consistent license headers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

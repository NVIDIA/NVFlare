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

"""Point a release branch's web pages at the documentation of its latest release.

Read the Docs publishes documentation per release (for example ``2.9.0``), not
per release branch (``2.9``), so links built from the branch name return 404.
This script finds the newest built ``<branch>.<patch>`` version on Read the Docs
and rewrites the branch's documentation links in a built site directory to it.
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

RTD_VERSIONS_URL = "https://readthedocs.org/api/v3/projects/nvflare/versions/?active=true&limit=100"
DOCS_URL_PREFIX = "https://nvflare.readthedocs.io/en/"
REWRITTEN_SUFFIXES = (".html", ".js")


def built_docs_versions() -> list:
    """Return the slugs of all active Read the Docs versions whose docs are built."""
    slugs = []
    url = RTD_VERSIONS_URL
    while url:
        with urllib.request.urlopen(url, timeout=30) as response:
            page = json.load(response)
        slugs.extend(version["slug"] for version in page["results"] if version.get("built"))
        url = page.get("next")
    return slugs


def latest_release(branch: str, slugs: list):
    """Return the highest ``<branch>.<patch>`` slug, or None if the branch has no built release."""
    pattern = re.compile(rf"^{re.escape(branch)}\.(\d+)$")
    patches = [int(match.group(1)) for match in map(pattern.match, slugs) if match]
    return f"{branch}.{max(patches)}" if patches else None


def rewrite_docs_links(site_dir: Path, branch: str, docs_version: str) -> int:
    """Rewrite links to the branch's docs; return the number of files changed.

    Matches ``<branch>/`` and ``<branch>.<patch>/``, plus the unexpanded
    ``${gh_branch}/`` and ``{gh_branch}/`` left in pages built from a quoting bug.
    Version-root URLs may omit the trailing slash or end in a query/fragment.
    Unreadable files are left unchanged with a warning so other pages still update.
    """
    link = re.compile(
        rf"{re.escape(DOCS_URL_PREFIX)}(?:{re.escape(branch)}(?:\.\d+)?|\$?\{{gh_branch\}})" r"(?=/|[?#\"'`\s<>]|$)"
    )
    replacement = f"{DOCS_URL_PREFIX}{docs_version}"
    changed = 0
    for path in site_dir.rglob("*"):
        if not path.is_file() or path.suffix not in REWRITTEN_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as e:
            print(f"::warning::Could not read {path} ({e}); file left unchanged")
            continue
        updated = link.sub(replacement, text)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--branch", required=True, help="release branch, for example 2.9")
    parser.add_argument("--site-dir", required=True, type=Path, help="built site directory for that branch")
    args = parser.parse_args()

    if not args.site_dir.is_dir():
        parser.error(f"--site-dir {args.site_dir} is not a directory")
    # Warn and skip instead of failing: a skipped run leaves the pages as built,
    # and the next scheduled refresh retries.
    if not re.fullmatch(r"\d+\.\d+", args.branch):
        print(f"::warning::{args.branch!r} is not a release branch; documentation links left unchanged")
        return 0
    try:
        slugs = built_docs_versions()
    except (OSError, ValueError, KeyError) as e:
        print(f"::warning::Could not list Read the Docs versions ({e}); {args.branch} links left unchanged")
        return 0

    docs_version = latest_release(args.branch, slugs)
    if docs_version is None:
        # Leave the links unchanged so a later run can still find and update them
        # once the branch's first release is built.
        print(f"::warning::No built {args.branch}.x release on Read the Docs; {args.branch} links left unchanged")
        return 0

    changed = rewrite_docs_links(args.site_dir, args.branch, docs_version)
    print(f"{args.branch}: documentation links point to {docs_version} ({changed} files updated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

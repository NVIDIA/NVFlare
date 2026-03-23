#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION.
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

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse


MARKDOWN_LINK_RE = re.compile(r"(?<!\!)\[[^\]]*\]\(\s*([^)]+?)\s*\)")
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(\s*([^)]+?)\s*\)")
MARKDOWN_REFERENCE_DEF_RE = re.compile(r"(?m)^\s*\[([^\]]+)\]:\s*(\S.*)$")
MARKDOWN_REFERENCE_USE_RE = re.compile(r"(?<!\!)\[([^\]]+)\]\[([^\]]*)\]")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
MARKDOWN_FENCE_OPEN_RE = re.compile(r"^(`{3,}|~{3,}).*$")
SUPPORTED_EXTENSIONS = {".md", ".html"}


@dataclass(frozen=True)
class LinkProblem:
    file_path: Path
    line: int
    target: str
    message: str


def _mask_html_comments(text: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        return "\n" * match.group(0).count("\n")

    return HTML_COMMENT_RE.sub(replacer, text)


def _mask_markdown_fences(text: str) -> str:
    masked_lines = []
    fence_marker = None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if fence_marker is not None:
            if re.match(rf"^{re.escape(fence_marker[0])}{{{len(fence_marker)},}}\s*$", stripped):
                fence_marker = None
            masked_lines.append("\n" if line.endswith("\n") else "")
            continue

        match = MARKDOWN_FENCE_OPEN_RE.match(stripped)
        if match:
            fence_marker = match.group(1)
            masked_lines.append("\n" if line.endswith("\n") else "")
            continue

        masked_lines.append(line)
    return "".join(masked_lines)


def _line_number(text: str, start_index: int) -> int:
    return text.count("\n", 0, start_index) + 1


def _normalize_markdown_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<"):
        end = target.find(">")
        return target[1:end] if end != -1 else target
    parts = target.split()
    return parts[0] if parts else ""


def _clean_target(raw_target: str) -> str | None:
    target = raw_target.strip()
    if not target or target.startswith("#") or target.startswith("//"):
        return None

    target = target.split("#", 1)[0].split("?", 1)[0].strip()
    if not target:
        return None

    parsed = urlparse(target)
    if parsed.scheme:
        return None

    return unquote(target)


def _resolve_path(target: str, file_path: Path, repo_root: Path) -> tuple[Path | None, str | None]:
    if target.startswith("/"):
        resolved = (repo_root / target.lstrip("/")).resolve()
    else:
        resolved = (file_path.parent / target).resolve()

    try:
        resolved.relative_to(repo_root)
    except ValueError:
        return None, "resolves outside the repository"

    if not resolved.exists():
        return None, "target does not exist"

    return resolved, None


class _HTMLLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.targets: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.targets.append((self.getpos()[0], value))


def _extract_markdown_targets(text: str) -> list[tuple[int, str]]:
    masked = _mask_html_comments(_mask_markdown_fences(text))
    targets = []

    reference_definitions: dict[str, str] = {}
    for label, target in MARKDOWN_REFERENCE_DEF_RE.findall(masked):
        normalized_label = label.strip().lower()
        normalized_target = _normalize_markdown_target(target)
        reference_definitions.setdefault(normalized_label, normalized_target)

    for pattern in (MARKDOWN_LINK_RE, MARKDOWN_IMAGE_RE):
        for match in pattern.finditer(masked):
            targets.append((_line_number(masked, match.start()), _normalize_markdown_target(match.group(1))))

    for match in MARKDOWN_REFERENCE_USE_RE.finditer(masked):
        label = (match.group(2) or match.group(1)).strip().lower()
        target = reference_definitions.get(label)
        if target:
            targets.append((_line_number(masked, match.start()), target))

    parser = _HTMLLinkParser()
    parser.feed(masked)
    targets.extend(parser.targets)
    return targets


def _extract_html_targets(text: str) -> list[tuple[int, str]]:
    parser = _HTMLLinkParser()
    parser.feed(_mask_html_comments(text))
    return parser.targets


def check_relative_doc_links(paths: Iterable[Path], repo_root: Path) -> list[LinkProblem]:
    repo_root = repo_root.resolve()
    problems: list[LinkProblem] = []
    seen: set[tuple[Path, int, str]] = set()

    for path in paths:
        file_path = path.resolve()
        if file_path.suffix not in SUPPORTED_EXTENSIONS:
            continue

        text = file_path.read_text(encoding="utf-8")
        if file_path.suffix == ".md":
            targets = _extract_markdown_targets(text)
        else:
            targets = _extract_html_targets(text)

        for line, raw_target in targets:
            target = _clean_target(raw_target)
            if target is None:
                continue

            _, error = _resolve_path(target, file_path, repo_root)
            if error is None:
                continue

            key = (file_path, line, target)
            if key in seen:
                continue
            seen.add(key)
            problems.append(LinkProblem(file_path=file_path, line=line, target=target, message=error))

    return sorted(problems, key=lambda p: (str(p.file_path), p.line, p.target))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Markdown and HTML relative links inside the repository.")
    parser.add_argument("paths", nargs="*", type=Path, help="Files to validate.")
    parser.add_argument(
        "--files-from",
        type=Path,
        help="Optional newline-delimited file list. Empty lines are ignored.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve absolute-root links. Defaults to the current working directory.",
    )
    return parser.parse_args()


def _load_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.paths)
    if args.files_from:
        file_list = [
            Path(line.strip())
            for line in args.files_from.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        paths.extend(file_list)
    return paths


def main() -> int:
    args = _parse_args()
    paths = _load_paths(args)
    if not paths:
        return 0

    problems = check_relative_doc_links(paths, repo_root=args.repo_root)
    for problem in problems:
        print(f"{problem.file_path}:{problem.line}: {problem.target} ({problem.message})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

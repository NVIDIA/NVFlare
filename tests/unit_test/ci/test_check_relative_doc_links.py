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

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    script_path = Path(__file__).resolve().parents[3] / "ci" / "check_relative_doc_links.py"
    spec = importlib.util.spec_from_file_location("check_relative_doc_links", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return _load_module()


def test_detects_broken_markdown_relative_link(tmp_path, checker):
    repo_root = tmp_path
    docs_dir = repo_root / "docs"
    docs_dir.mkdir()
    readme = docs_dir / "README.md"
    readme.write_text("[broken](./missing.md)\n", encoding="utf-8")

    problems = checker.check_relative_doc_links([readme], repo_root)

    assert len(problems) == 1
    assert problems[0].line == 1
    assert problems[0].target == "./missing.md"


def test_ignores_literal_file_uri_examples_and_fenced_code(tmp_path, checker):
    repo_root = tmp_path
    docs_dir = repo_root / "docs"
    docs_dir.mkdir()
    target = docs_dir / "target.md"
    target.write_text("ok\n", encoding="utf-8")
    readme = docs_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "```python",
                "[code only](./missing.md)",
                "```",
                '`tracking_uri = "file:///tmp/mlruns"`',
                "[good](./target.md)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    problems = checker.check_relative_doc_links([readme], repo_root)

    assert problems == []


def test_fenced_block_stays_masked_when_inner_line_looks_like_opening_fence(tmp_path, checker):
    repo_root = tmp_path
    docs_dir = repo_root / "docs"
    docs_dir.mkdir()
    target = docs_dir / "target.md"
    target.write_text("ok\n", encoding="utf-8")
    readme = docs_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "```shell",
                "#!/bin/bash",
                "```python",
                "[broken](./missing.md)",
                "```",
                "```",
                "[good](./target.md)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    problems = checker.check_relative_doc_links([readme], repo_root)

    assert problems == []


def test_ignores_whitespace_only_markdown_target(tmp_path, checker):
    repo_root = tmp_path
    docs_dir = repo_root / "docs"
    docs_dir.mkdir()
    readme = docs_dir / "README.md"
    readme.write_text("[empty]( )\n", encoding="utf-8")

    problems = checker.check_relative_doc_links([readme], repo_root)

    assert problems == []


def test_checks_html_src_inside_markdown(tmp_path, checker):
    repo_root = tmp_path
    docs_dir = repo_root / "docs"
    figs_dir = docs_dir / "figs"
    figs_dir.mkdir(parents=True)
    (figs_dir / "plot.png").write_text("not-an-image-but-exists\n", encoding="utf-8")
    readme = docs_dir / "README.md"
    readme.write_text('<img src="./figs/plot.png" alt="plot"/>\n', encoding="utf-8")

    problems = checker.check_relative_doc_links([readme], repo_root)

    assert problems == []


def test_ignores_inline_code_spans_with_markdown_and_html_links(tmp_path, checker):
    repo_root = tmp_path
    docs_dir = repo_root / "docs"
    docs_dir.mkdir()
    target = docs_dir / "target.md"
    target.write_text("ok\n", encoding="utf-8")
    readme = docs_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "`[code-only](./missing.md)`",
                '`<img src="./missing.png" alt="inline-example"/>`',
                "[good](./target.md)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    problems = checker.check_relative_doc_links([readme], repo_root)

    assert problems == []


def test_duplicate_reference_definitions_keep_first_match(tmp_path, checker):
    repo_root = tmp_path
    docs_dir = repo_root / "docs"
    docs_dir.mkdir()
    target = docs_dir / "target.md"
    target.write_text("ok\n", encoding="utf-8")
    readme = docs_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "[ref-link][shared]",
                "",
                "[shared]: ./missing.md",
                "[shared]: ./target.md",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    problems = checker.check_relative_doc_links([readme], repo_root)

    assert len(problems) == 1
    assert problems[0].target == "./missing.md"

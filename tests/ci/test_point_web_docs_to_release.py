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
import io
import json
from pathlib import Path
from urllib.error import URLError

import pytest


@pytest.fixture(scope="module")
def docs_links():
    script = Path(__file__).resolve().parents[2] / ".github/scripts/point_web_docs_to_release.py"
    spec = importlib.util.spec_from_file_location("point_web_docs_to_release", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selects_highest_numeric_built_patch(docs_links, monkeypatch):
    pages = {
        docs_links.RTD_VERSIONS_URL: {
            "results": [
                {"slug": "2.9.9", "built": True},
                {"slug": "2.9.11", "built": False},
                {"slug": "2.9.12"},
            ],
            "next": "https://example.test/page2",
        },
        "https://example.test/page2": {
            "results": [
                {"slug": "2.9.10", "built": True},
                {"slug": "2.9.20rc1", "built": True},
                {"slug": "2.10.0", "built": True},
                {"slug": "main", "built": True},
            ],
            "next": None,
        },
    }
    seen = []

    def urlopen(url, timeout):
        seen.append(url)
        return io.StringIO(json.dumps(pages[url]))

    monkeypatch.setattr(docs_links.urllib.request, "urlopen", urlopen)
    slugs = docs_links.built_docs_versions()
    assert seen == list(pages)
    assert docs_links.latest_release("2.9", slugs) == "2.9.10"
    assert docs_links.latest_release("2.8", slugs) is None


@pytest.mark.parametrize("suffix", ["/guide.html", "", "?view=1", "#intro"])
@pytest.mark.parametrize("version", ["2.9", "2.9.0", "${gh_branch}", "{gh_branch}"])
def test_rewrites_links_and_is_idempotent(docs_links, tmp_path, suffix, version):
    page = tmp_path / "index.html"
    page.write_text(f'<a href="{docs_links.DOCS_URL_PREFIX}{version}{suffix}">docs</a>', encoding="utf-8")
    assert docs_links.rewrite_docs_links(tmp_path, "2.9", "2.9.10") == 1
    assert page.read_text() == f'<a href="{docs_links.DOCS_URL_PREFIX}2.9.10{suffix}">docs</a>'
    assert docs_links.rewrite_docs_links(tmp_path, "2.9", "2.9.10") == 0


def test_preserves_other_versions_and_non_site_files(docs_links, tmp_path):
    versions = ["main", "2.8", "2.8.2", "2.90", "2.9.1rc1", "2.9.example"]
    text = " ".join(f"{docs_links.DOCS_URL_PREFIX}{v}/index.html" for v in versions)
    (tmp_path / "index.html").write_text(text, encoding="utf-8")
    ignored = tmp_path / "notes.txt"
    ignored.write_text(f"{docs_links.DOCS_URL_PREFIX}2.9/index.html", encoding="utf-8")
    assert docs_links.rewrite_docs_links(tmp_path, "2.9", "2.9.10") == 0
    assert (tmp_path / "index.html").read_text() == text
    assert ignored.read_text() == f"{docs_links.DOCS_URL_PREFIX}2.9/index.html"


def test_rewrites_nested_javascript_and_bare_root(docs_links, tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    script = assets / "links.js"
    script.write_text(f"{docs_links.DOCS_URL_PREFIX}2.9", encoding="utf-8")
    assert docs_links.rewrite_docs_links(tmp_path, "2.9", "2.9.10") == 1
    assert script.read_text() == f"{docs_links.DOCS_URL_PREFIX}2.9.10"


def test_bad_encoding_does_not_stop_other_pages(docs_links, tmp_path, capsys):
    bad = tmp_path / "bad.html"
    bad.write_bytes(b"\xff\xfe")
    good = tmp_path / "good.html"
    good.write_text(f"{docs_links.DOCS_URL_PREFIX}2.9/index.html", encoding="utf-8")
    assert docs_links.rewrite_docs_links(tmp_path, "2.9", "2.9.10") == 1
    assert bad.read_bytes() == b"\xff\xfe"
    assert good.read_text() == f"{docs_links.DOCS_URL_PREFIX}2.9.10/index.html"
    assert "::warning::Could not read" in capsys.readouterr().out


def test_vanished_file_does_not_stop_other_pages(docs_links, tmp_path, monkeypatch, capsys):
    missing = tmp_path / "missing.html"
    missing.write_text("gone", encoding="utf-8")
    good = tmp_path / "good.html"
    good.write_text(f"{docs_links.DOCS_URL_PREFIX}2.9/index.html", encoding="utf-8")
    read_text = Path.read_text

    def read_or_disappear(path, *args, **kwargs):
        if path == missing:
            raise FileNotFoundError("file vanished after discovery")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_or_disappear)
    assert docs_links.rewrite_docs_links(tmp_path, "2.9", "2.9.10") == 1
    assert good.read_text() == f"{docs_links.DOCS_URL_PREFIX}2.9.10/index.html"
    assert "file vanished" in capsys.readouterr().out


@pytest.mark.parametrize("outcome", [URLError("offline"), [], ["2.8.2"]])
def test_unavailable_release_leaves_links_unchanged(docs_links, tmp_path, monkeypatch, capsys, outcome):
    page = tmp_path / "index.html"
    original = f"{docs_links.DOCS_URL_PREFIX}2.9/index.html"
    page.write_text(original, encoding="utf-8")

    def versions():
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(docs_links, "built_docs_versions", versions)
    monkeypatch.setattr("sys.argv", ["rewrite", "--branch", "2.9", "--site-dir", str(tmp_path)])
    assert docs_links.main() == 0
    assert page.read_text() == original
    assert "::warning::" in capsys.readouterr().out


def test_invalid_branch_never_queries_api(docs_links, tmp_path, monkeypatch, capsys):
    def unexpected_query():
        pytest.fail("invalid branch must be rejected before querying Read the Docs")

    monkeypatch.setattr(docs_links, "built_docs_versions", unexpected_query)
    monkeypatch.setattr("sys.argv", ["rewrite", "--branch", "feature/x", "--site-dir", str(tmp_path)])
    assert docs_links.main() == 0
    assert "not a release branch" in capsys.readouterr().out

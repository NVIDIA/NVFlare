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

import json
from pathlib import Path

import pytest

from nvflare.tool.examples.catalog import load_catalog

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_catalog_entries_have_category_and_source_path_and_exist():
    catalog, errors = load_catalog()

    assert errors == []
    assert catalog["cifar10-pt"] == {
        "category": "advanced",
        "source_path": "examples/advanced/cifar10/pt",
    }
    for entry in catalog.values():
        assert set(entry) == {"category", "source_path"}
        assert not entry["source_path"].startswith("examples/tutorials/")
        source = REPO_ROOT / entry["source_path"]
        assert source.is_dir()
        assert (source / "README.md").is_file() or (source / "README.rst").is_file()
        assert any(
            path.is_file() and path.name not in {"README.md", "README.rst"} for path in source.rglob("*")
        ), f"{entry['source_path']} contains no example files"


def test_catalog_excludes_examples_that_require_files_outside_the_downloaded_subtree():
    catalog, _ = load_catalog()
    source_paths = {entry["source_path"] for entry in catalog.values()}

    assert "examples/advanced/cifar10/pt" in source_paths
    assert source_paths.isdisjoint(
        {
            "examples/advanced/cifar10/pt/cifar10-real-world",
            "examples/advanced/cifar10/pt/cifar10-sim",
            "examples/advanced/collab/pt_async_cifar10",
            "examples/advanced/collab/pt_cifar10",
            "examples/advanced/hello-pt-environments",
            "examples/docker",
            "examples/devops/multicloud",
            "examples/devops/openshift",
            "examples/hello-world/agent-skills/fedstats-image",
            "examples/hello-world/agent-skills/fedstats-tabular",
            "examples/hello-world/agent-skills/huggingface-conversion",
            "examples/hello-world/agent-skills/lightning-conversion",
            "examples/hello-world/agent-skills/pytorch-conversion",
        }
    )


@pytest.mark.parametrize(
    "invalid_entry",
    [
        {"bad name": {"category": "test", "source_path": "examples/demo"}},
        {"demo": {"category": "test", "source_path": "outside/demo"}},
        {"demo": {"category": "test", "source_path": "examples/../demo"}},
        {"demo": {"category": "test", "source_path": "examples/demo/"}},
        {"demo": {"source_path": "examples/demo"}},
        {"demo": {"category": "Bad Category", "source_path": "examples/demo"}},
        {
            "demo": {
                "category": "test",
                "source_path": "examples/demo",
                "next_command": ["python", "job.py"],
            }
        },
        {"duplicate": {"category": "test", "source_path": "examples/good"}},
    ],
)
def test_invalid_entry_does_not_hide_valid_entries(tmp_path, invalid_entry):
    path = tmp_path / "catalog.json"
    definitions = {"good": {"category": "test", "source_path": "examples/good"}, **invalid_entry}
    path.write_text(json.dumps(definitions))

    catalog, errors = load_catalog(path)

    assert catalog == {"good": {"category": "test", "source_path": "examples/good"}}
    assert len(errors) == 1


@pytest.mark.parametrize("contents", ["{}", "[]", "not JSON"])
def test_invalid_catalog_document_is_rejected(tmp_path, contents):
    path = tmp_path / "catalog.json"
    path.write_text(contents)

    with pytest.raises(ValueError):
        load_catalog(path)


def test_duplicate_short_name_is_rejected(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text('{"duplicate":{"source_path":"examples/one"},"duplicate":{"source_path":"examples/two"}}')

    with pytest.raises(ValueError, match="duplicate catalog short name"):
        load_catalog(path)


def test_duplicate_entry_key_does_not_hide_valid_entries(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(
        '{"good":{"category":"test","source_path":"examples/good"},'
        '"duplicate":{"category":"test","source_path":"examples/one","source_path":"examples/two"}}'
    )

    catalog, errors = load_catalog(path)

    assert catalog == {"good": {"category": "test", "source_path": "examples/good"}}
    assert errors == [{"name": "duplicate", "error": "contains duplicate key source_path"}]

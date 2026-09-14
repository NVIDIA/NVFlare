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


def test_catalog_entries_are_source_path_only_and_exist():
    catalog, errors = load_catalog()

    assert errors == []
    assert catalog["collab-pt"] == {"source_path": "examples/advanced/collab/pt_cifar10"}
    for entry in catalog.values():
        assert set(entry) == {"source_path"}
        assert not entry["source_path"].startswith("examples/tutorials/")
        source = REPO_ROOT / entry["source_path"]
        assert source.is_dir()
        assert (source / "README.md").is_file() or (source / "README.rst").is_file()


def test_catalog_covers_each_example_collection():
    catalog, _ = load_catalog()
    source_paths = [Path(entry["source_path"]) for entry in catalog.values()]
    collections = [REPO_ROOT / "examples/hello-world", REPO_ROOT / "examples/advanced"]
    collections.extend(
        [
            REPO_ROOT / "examples/devops/aws",
            REPO_ROOT / "examples/devops/azure",
            REPO_ROOT / "examples/devops/gcp",
        ]
    )
    for collection in collections:
        for example in (path for path in collection.iterdir() if path.is_dir()):
            relative = example.relative_to(REPO_ROOT)
            assert any(path == relative or relative in path.parents for path in source_paths), relative

    for relative in ["examples/docker", "examples/devops/multicloud", "examples/devops/openshift"]:
        assert Path(relative) in source_paths


@pytest.mark.parametrize(
    "invalid_entry",
    [
        {"bad name": {"source_path": "examples/demo"}},
        {"demo": {"source_path": "outside/demo"}},
        {"demo": {"source_path": "examples/../demo"}},
        {"demo": {"source_path": "examples/demo/"}},
        {"demo": {"source_path": "examples/demo", "next_command": ["python", "job.py"]}},
        {"duplicate": {"source_path": "examples/good"}},
    ],
)
def test_invalid_entry_does_not_hide_valid_entries(tmp_path, invalid_entry):
    path = tmp_path / "catalog.json"
    definitions = {"good": {"source_path": "examples/good"}, **invalid_entry}
    path.write_text(json.dumps(definitions))

    catalog, errors = load_catalog(path)

    assert catalog == {"good": {"source_path": "examples/good"}}
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
        '{"good":{"source_path":"examples/good"},'
        '"duplicate":{"source_path":"examples/one","source_path":"examples/two"}}'
    )

    catalog, errors = load_catalog(path)

    assert catalog == {"good": {"source_path": "examples/good"}}
    assert errors == [{"name": "duplicate", "error": "contains duplicate key source_path"}]

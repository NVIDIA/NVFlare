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
import subprocess
from pathlib import Path

import pytest

from nvflare.tool.examples.catalog import load_catalog

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_catalog_entries_have_category_and_source_path_and_exist():
    catalog = load_catalog()
    for entry in catalog.values():
        assert {"category", "source_path"} <= set(entry) <= {"category", "source_path", "destination_path"}
        assert not entry["source_path"].startswith("examples/tutorials/")
        source = REPO_ROOT / entry["source_path"]
        assert source.is_dir()
        assert (source / "README.md").is_file() or (source / "README.rst").is_file()
        assert any(
            path.is_file() and path.name not in {"README.md", "README.rst"} for path in source.rglob("*")
        ), f"{entry['source_path']} contains no example files"


def test_catalog_covers_every_maintained_example_collection():
    catalog = load_catalog()
    source_paths = {Path(entry["source_path"]) for entry in catalog.values()}
    assert len(source_paths) == len(catalog)
    try:
        tracked_output = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "--", "examples"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("catalog coverage requires an NVFlare Git checkout")
    tracked_files = {Path(path) for path in tracked_output.splitlines()}
    tracked_readme_dirs = {path.parent for path in tracked_files if path.name in {"README.md", "README.rst"}}

    split_collections = {
        Path("examples/hello-world/agent-skills"),
        Path("examples/advanced/cifar10"),
        Path("examples/advanced/collab"),
        Path("examples/advanced/federated-statistics"),
        Path("examples/advanced/job_api"),
        Path("examples/advanced/monai"),
        Path("examples/advanced/multi-gpu"),
        Path("examples/advanced/vertical_federated_learning"),
    }
    excluded = {
        Path("examples/advanced/finance"),
        Path("examples/advanced/finance-end-to-end"),
        Path("examples/advanced/hello-pt-environments"),
        Path("examples/advanced/nlp-ner"),
    }
    assert excluded <= tracked_readme_dirs
    expected_source_paths = set()
    for parent in (Path("examples/hello-world"), Path("examples/advanced")):
        for relative in sorted(path for path in tracked_readme_dirs if path.parent == parent):
            if relative in excluded:
                continue
            if relative in split_collections:
                expected_source_paths.update(path for path in tracked_readme_dirs if path.parent == relative)
            else:
                expected_source_paths.add(relative)

    devops = Path("examples/devops")
    devops_collections = {
        Path(*path.parts[:3]) for path in tracked_files if len(path.parts) > 3 and path.parts[:2] == devops.parts
    }
    for collection in sorted(devops_collections):
        if collection.name in {"aws", "azure", "gcp"}:
            expected_source_paths.update(path for path in tracked_readme_dirs if path.parent == collection)
        elif collection in tracked_readme_dirs:
            expected_source_paths.add(collection)
    expected_source_paths.add(Path("examples/docker"))

    # This exact comparison checks both membership and count. A new maintained
    # example must be cataloged, and a stale catalog entry cannot linger after
    # its source is removed.
    assert source_paths == expected_source_paths
    assert len(catalog) == len(expected_source_paths)


@pytest.mark.parametrize(
    "invalid_entry",
    [
        {"bad name": {"category": "test", "source_path": "examples/demo"}},
        {"demo": {"category": "test", "source_path": "examples"}},
        {"demo": {"category": "test", "source_path": "outside/demo"}},
        {"demo": {"category": "test", "source_path": "examples/../demo"}},
        {"demo": {"category": "test", "source_path": "examples/demo/"}},
        {"demo": {"category": "test", "source_path": "examples/de\x00mo"}},
        {"demo": {"category": "test", "source_path": "examples/demo", "destination_path": "/demo"}},
        {"demo": {"category": "test", "source_path": "examples/demo", "destination_path": "../demo"}},
        {"demo": {"category": "test", "source_path": "examples/demo", "destination_path": "demo/"}},
        {"demo": {"category": "test", "source_path": "examples/demo", "destination_path": "de\x00mo"}},
        {
            "demo": {
                "category": "test",
                "source_path": "examples/demo",
                "destination_path": ".nvflare-example.json",
            }
        },
        {
            "demo": {
                "category": "test",
                "source_path": "examples/demo",
                "destination_path": ".NVFLARE-EXAMPLE.JSON/nested",
            }
        },
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
def test_invalid_entry_rejects_catalog(tmp_path, invalid_entry):
    path = tmp_path / "catalog.json"
    definitions = {"good": {"category": "test", "source_path": "examples/good"}, **invalid_entry}
    path.write_text(json.dumps(definitions))

    with pytest.raises(ValueError, match="invalid catalog entry"):
        load_catalog(path)


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


def test_duplicate_entry_key_rejects_catalog(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(
        '{"good":{"category":"test","source_path":"examples/good"},'
        '"duplicate":{"category":"test","source_path":"examples/one","source_path":"examples/two"}}'
    )

    with pytest.raises(ValueError, match="contains duplicate key source_path"):
        load_catalog(path)

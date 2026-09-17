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
    assert "examples/advanced/experiment-tracking" in source_paths
    assert (REPO_ROOT / "examples/advanced/experiment-tracking/prepare_data.sh").is_file()
    assert "examples/hello-world/agent-skills/pytorch-conversion" in source_paths
    assert source_paths.isdisjoint(
        {
            "examples/advanced/cifar10/pt/cifar10-real-world",
            "examples/advanced/cifar10/pt/cifar10-sim",
            "examples/advanced/collab/pt_async_cifar10",
            "examples/advanced/collab/pt_cifar10",
            "examples/advanced/experiment-tracking/mlflow",
            "examples/advanced/experiment-tracking/tensorboard",
            "examples/advanced/experiment-tracking/wandb",
            "examples/advanced/hello-pt-environments",
            "examples/docker",
            "examples/devops/multicloud",
            "examples/devops/openshift",
        }
    )


def test_experiment_tracking_quickstart_uses_downloaded_layout():
    readme = (REPO_ROOT / "examples" / "advanced" / "experiment-tracking" / "README.md").read_text(encoding="utf-8")

    assert "not installed by `nvflare examples get`" in readme
    assert "python -m pip install mlflow" in readme
    assert "python -m pip install tensorboard" in readme
    assert "python -m pip install wandb" in readme
    assert "skip their\n**Install Requirements** steps" in readme
    assert "pip install -r requirements.txt" not in readme
    assert "./prepare_data.sh" in readme
    assert "cd tensorboard" in readme
    assert "cd wandb" in readme
    assert "cd mlflow/<example-name>" in readme
    assert "cd examples/advanced/experiment-tracking/<framework>" not in readme
    assert "<framework>/jobs/<job_name>/code" not in readme


def test_huggingface_guidance_preserves_installed_distribution():
    example = REPO_ROOT / "examples" / "hello-world" / "hello-huggingface"
    readme = (example / "README.md").read_text(encoding="utf-8")
    requirements = (example / "requirements.txt").read_text(encoding="utf-8")

    assert 'python -m pip install "nvflare[PT]"' in readme
    assert 'python -m pip install "nvflare-nightly[PT]"' in readme
    assert 'python -m pip install -e ".[PT]"' in readme
    assert "nvflare" not in requirements.casefold()


def test_hello_pt_guidance_preserves_revision_for_install_and_environment_follow_up():
    example_readme = (REPO_ROOT / "examples" / "hello-world" / "hello-pt" / "README.md").read_text(encoding="utf-8")
    advanced_readme = (REPO_ROOT / "examples" / "advanced" / "hello-pt-environments" / "README.md").read_text(
        encoding="utf-8"
    )
    docs_page = (REPO_ROOT / "docs" / "hello-world" / "hello-pt" / "index.rst").read_text(encoding="utf-8")

    assert 'python -m pip install "nvflare[PT]"' in docs_page
    assert "nvflare examples get hello-pt" in docs_page
    assert 'python -m pip install -e ".[PT]"' in docs_page
    assert "install that checkout in\neditable mode" in docs_page
    assert "NVFLARE_REVISION=$(nvflare examples revision)" in example_readme
    assert "tree/%s/examples/advanced/hello-pt-environments" in example_readme
    assert 'git -C ../nvflare-source checkout "$NVFLARE_REVISION"' in example_readme
    assert "../../advanced/hello-pt-environments/README.md" not in example_readme
    assert 'python -m pip install -e ".[PT]"' in advanced_readme
    assert "python -m pip install -r requirements.txt" not in advanced_readme


@pytest.mark.parametrize(
    "example_name",
    [
        "fedstats-image",
        "fedstats-tabular",
        "huggingface-conversion",
        "lightning-conversion",
        "pytorch-conversion",
    ],
)
def test_agent_skill_examples_install_skills_from_repository(example_name):
    readme = (REPO_ROOT / "examples" / "hello-world" / "agent-skills" / example_name / "README.md").read_text(
        encoding="utf-8"
    )

    assert "https://github.com/NVIDIA/NVFlare/tree/${NVFLARE_REVISION}/skills" in readme
    assert "nvflare examples revision" in readme
    assert '"<nvflare-repo>/skills"' in readme
    assert "../../../../skills" not in readme
    assert "pip install 'nvflare" not in readme


@pytest.mark.parametrize(
    "invalid_entry",
    [
        {"bad name": {"category": "test", "source_path": "examples/demo"}},
        {"demo": {"category": "test", "source_path": "examples"}},
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

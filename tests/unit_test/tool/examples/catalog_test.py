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
    catalog, errors = load_catalog()

    assert errors == []
    assert catalog["cifar10-pt"] == {
        "category": "advanced",
        "source_path": "examples/advanced/cifar10/pt",
    }
    assert catalog["collab-pt"] == {
        "category": "advanced",
        "source_path": "examples/advanced/collab/pt_cifar10",
        "destination_path": "collab/pt_cifar10",
    }
    assert catalog["collab-pt-async"] == {
        "category": "advanced",
        "source_path": "examples/advanced/collab/pt_async_cifar10",
        "destination_path": "collab/pt_async_cifar10",
    }
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
    catalog, _ = load_catalog()
    source_paths = {Path(entry["source_path"]) for entry in catalog.values()}
    assert len(source_paths) == len(catalog)
    tracked_files = {
        Path(path)
        for path in subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "--", "examples"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    }
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

    assert Path("examples/advanced/cifar10/pt") in source_paths
    assert Path("examples/advanced/collab/pt_cifar10") in source_paths
    assert Path("examples/advanced/collab/pt_async_cifar10") in source_paths
    assert Path("examples/advanced/experiment-tracking") in source_paths
    assert (REPO_ROOT / "examples/advanced/experiment-tracking/prepare_data.sh").is_file()
    assert Path("examples/hello-world/agent-skills/pytorch-conversion") in source_paths
    assert source_paths.isdisjoint(
        {
            Path("examples/advanced/cifar10/pt/cifar10-real-world"),
            Path("examples/advanced/cifar10/pt/cifar10-sim"),
            Path("examples/advanced/experiment-tracking/mlflow"),
            Path("examples/advanced/experiment-tracking/tensorboard"),
            Path("examples/advanced/experiment-tracking/wandb"),
        }
    )


def test_collab_pt_quickstart_uses_downloaded_package_layout():
    readme = (REPO_ROOT / "examples" / "advanced" / "collab" / "pt_cifar10" / "README.md").read_text(encoding="utf-8")

    assert "nvflare examples get collab-pt" in readme
    assert "python -m pip install -r collab/pt_cifar10/requirements.txt" in readme
    assert "python collab/pt_cifar10/prepare_data.py" in readme
    assert "python -m collab.pt_cifar10.fedavg.job" in readme


def test_collab_pt_async_is_standalone_and_uses_downloaded_package_layout():
    example = REPO_ROOT / "examples" / "advanced" / "collab" / "pt_async_cifar10"
    readme = (example / "README.md").read_text(encoding="utf-8")
    job = (example / "job.py").read_text(encoding="utf-8")
    prepare = (example / "prepare_data.sh").read_text(encoding="utf-8")

    assert "nvflare examples get collab-pt-async" in readme
    assert "cd collab-pt-async/collab/pt_async_cifar10" in readme
    assert "from cifar10_data import split_and_save" in job
    assert "../../cifar10" not in prepare
    assert (example / "cifar10_data.py").is_file()


@pytest.mark.parametrize(
    "name,source_path,destination_path",
    [
        ("devops-aws-eks", "examples/devops/aws/eks", "aws/eks"),
        ("devops-azure-aks", "examples/devops/azure/aks", "azure/aks"),
        ("devops-gcp-gke", "examples/devops/gcp/gke", "gcp/gke"),
    ],
)
def test_deployment_downloads_preserve_script_directory_depth(name, source_path, destination_path):
    catalog, _ = load_catalog()
    entry = catalog[name]
    readme = (REPO_ROOT / source_path / "README.md").read_text(encoding="utf-8")
    script = (REPO_ROOT / source_path / "create_cluster.sh").read_text(encoding="utf-8")

    assert entry["destination_path"] == destination_path
    assert "${SCRIPT_DIR}/../.." in script
    assert f"nvflare examples get {name}" in readme
    assert f"cd {name}/{destination_path}" in readme


def test_docker_runtime_prepares_revision_matched_build_context():
    catalog, _ = load_catalog()
    example = REPO_ROOT / "examples" / "docker"
    script = (example / "build_docker.sh").read_text(encoding="utf-8")
    readme = (example / "README.md").read_text(encoding="utf-8")

    assert catalog["docker-runtime"]["destination_path"] == "examples/docker"
    assert 'nvflare examples revision --dir "$REPO_ROOT"' in script
    assert "$REPO_ROOT/.nvflare-example.json" in script
    assert 'version = json.load(f).get("nvflare_version")' in script
    assert 'git -C "$TEMP_CHECKOUT/repository" fetch --quiet --depth=1 origin "$REVISION"' in script
    assert 'git -C "$TEMP_CHECKOUT/repository" archive FETCH_HEAD' in script
    assert '"$BUILD_CONTEXT"' in script
    assert "nvflare examples get docker-runtime" in readme
    assert "cd docker-runtime/examples/docker" in readme


def test_multicloud_prepares_revision_matched_build_context():
    catalog, _ = load_catalog()
    example = REPO_ROOT / "examples" / "devops" / "multicloud"
    script = (example / "build_and_push.py").read_text(encoding="utf-8")
    readme = (example / "README.md").read_text(encoding="utf-8")

    assert catalog["devops-multicloud"]["destination_path"] == "examples/devops/multicloud"
    assert 'PROVENANCE_FILE = ".nvflare-example.json"' in script
    assert "revision, nvflare_base_version = downloaded_source_info()" in script
    assert 'f"NVFL_BASE_VERSION={nvflare_base_version}"' in script
    assert "nvflare examples get devops-multicloud" in readme
    assert "cd devops-multicloud" in readme


def test_openshift_download_contains_image_and_job_dependencies():
    catalog, _ = load_catalog()
    example = REPO_ROOT / "examples" / "devops" / "openshift"
    common = (example / "scripts" / "k8s_common.sh").read_text(encoding="utf-8")
    builder = (example / "scripts" / "build_images.sh").read_text(encoding="utf-8")
    readme = (example / "README.md").read_text(encoding="utf-8")

    assert catalog["devops-openshift"]["destination_path"] == "examples/devops/openshift"
    assert "nvflare examples get devops-openshift" in readme
    assert 'nvflare examples revision --dir "$DOWNLOAD_ROOT"' in builder
    assert "$DOWNLOAD_ROOT/.nvflare-example.json" in builder
    assert 'version = json.load(f).get("nvflare_version")' in builder
    assert "git clone --quiet --filter=blob:none --no-checkout" in builder
    assert 'git -C "$TEMP_SOURCE/source" checkout --quiet --detach FETCH_HEAD' in builder
    assert "git archive" not in builder
    assert 'client_script = pathlib.Path(example_root) / "jobs" / "numpy_client.py"' in common
    assert "hello-world/hello-numpy" not in common
    assert (example / "jobs" / "numpy_client.py").is_file()


def test_experiment_tracking_quickstart_uses_downloaded_layout():
    readme = (REPO_ROOT / "examples" / "advanced" / "experiment-tracking" / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())

    assert "not installed by `nvflare examples get`" in readme
    assert "python -m pip install mlflow" in readme
    assert "python -m pip install tensorboard" in readme
    assert "python -m pip install wandb" in readme
    assert "skip their **Install Requirements** steps" in normalized_readme
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


@pytest.mark.parametrize(
    "name,documentation",
    [
        ("hello-flower", "docs/hello-world/hello-flower/index.rst"),
        ("hello-jax", "docs/hello-world/hello-jax/index.rst"),
        ("hello-numpy", "docs/examples/hello_numpy.rst"),
        ("hello-tf", "docs/hello-world/hello-tf/index.rst"),
    ],
)
def test_downloaded_and_source_examples_preserve_the_selected_nvflare_distribution(name, documentation):
    example = REPO_ROOT / "examples" / "hello-world" / name
    readme = (example / "README.md").read_text(encoding="utf-8")
    docs_page = (REPO_ROOT / documentation).read_text(encoding="utf-8")
    requirements = (example / "requirements.txt").read_text(encoding="utf-8")

    for text in (readme, docs_page):
        assert "python -m pip install nvflare" in text
        assert "python -m pip install nvflare-nightly" in text
        assert f"nvflare examples get {name}" in text
        assert "python -m pip install -e ." in text
        assert "python -m pip install -r requirements.txt" in text
    assert "nvflare" not in requirements.casefold()


def test_flower_projects_use_the_parent_nvflare_distribution():
    example = REPO_ROOT / "examples" / "hello-world" / "hello-flower"

    for project in ("flwr-pt", "flwr-pt-tb"):
        pyproject = (example / project / "pyproject.toml").read_text(encoding="utf-8")
        assert '\n    "nvflare' not in pyproject.casefold()


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


def test_hello_lightning_guidance_preserves_installed_distribution():
    readme = (REPO_ROOT / "examples" / "hello-world" / "hello-lightning" / "README.md").read_text(encoding="utf-8")
    docs_page = (REPO_ROOT / "docs" / "hello-world" / "hello-lightning" / "index.rst").read_text(encoding="utf-8")

    for text in (readme, docs_page):
        assert 'python -m pip install "nvflare[PT]"' in text
        assert 'python -m pip install "nvflare-nightly[PT]"' in text
        assert 'python -m pip install -e ".[PT]"' in text
        assert "nvflare examples get hello-lightning" in text


def test_tracking_guidance_combines_required_extras():
    guide = (REPO_ROOT / "docs" / "user_guide" / "nvflare_cli" / "examples_command.rst").read_text(encoding="utf-8")

    assert 'python -m pip install "nvflare[PT,TRACKING]"' in guide
    assert 'python -m pip install "nvflare-nightly[PT,TRACKING]"' in guide
    assert 'python -m pip install -e ".[PT,TRACKING]"' in guide


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

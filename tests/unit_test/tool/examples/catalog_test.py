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

from pathlib import Path
from subprocess import CompletedProcess

from nvflare.tool.examples.catalog import load_catalog, source_files

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_catalog_entries_are_data_only_and_sources_exist():
    catalog = load_catalog()

    assert catalog["collab-pt"]["source_path"] == "examples/advanced/collab/pt_cifar10"
    for entry in catalog.values():
        assert "source_path" in entry
        assert set(entry) <= {"source_path", "prepare_commands", "next_command"}
        assert (REPO_ROOT / entry["source_path"]).is_dir()


def test_source_files_uses_git_manifest_for_checkout(monkeypatch, tmp_path):
    (tmp_path / ".git").touch()
    source = tmp_path / "examples/long-name"
    source.mkdir(parents=True)
    (source / "tracked.py").write_text("tracked")
    (source / "credentials.txt").write_text("untracked")

    def run(*args, **kwargs):
        return CompletedProcess(args[0], 0, stdout=b"examples/long-name/tracked.py\0")

    monkeypatch.setattr("nvflare.tool.examples.catalog.subprocess.run", run)

    assert source_files(tmp_path, "examples/long-name") == ("tracked.py",)


def test_source_files_walks_release_source_archive(tmp_path):
    source = tmp_path / "examples/long-name/nested"
    source.mkdir(parents=True)
    (source / "example.py").write_text("example")

    assert source_files(tmp_path, "examples/long-name") == ("nested/example.py",)

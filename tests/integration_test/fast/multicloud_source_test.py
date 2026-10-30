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
import os
import subprocess
import sys

from tests.git_test_utils import _git, _repo_root, _versioned_repository, require_signing_in_user_config


def _load_build_module():
    path = os.path.join(_repo_root(), "examples", "devops", "multicloud", "build_and_push.py")
    spec = importlib.util.spec_from_file_location("multicloud_build_and_push", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_revision_source_retains_git_metadata_for_versioneer(monkeypatch, tmp_path):
    require_signing_in_user_config(monkeypatch, tmp_path)
    build = _load_build_module()
    repository, revision = _versioned_repository(tmp_path)
    monkeypatch.setattr(build, "SOURCE_REPOSITORY", str(repository))

    temporary, source = build.prepare_revision_source(revision)
    try:
        assert _git(source, "rev-parse", "HEAD") == revision
        subprocess.run(
            [sys.executable, "-c", "import versioneer; assert not versioneer.get_versions()['error']"],
            cwd=source,
            check=True,
        )
    finally:
        temporary.cleanup()

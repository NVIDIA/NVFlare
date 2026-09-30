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
import os
import shutil
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_K8S_COMMON = _REPO_ROOT / "examples" / "devops" / "openshift" / "scripts" / "k8s_common.sh"
_BUILD_IMAGES = _K8S_COMMON.with_name("build_images.sh")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _write_prepare_config(tmp_path: Path, **resources: str) -> str:
    env = os.environ.copy()
    env.pop("PARENT_CPU", None)
    env.pop("PARENT_MEMORY", None)
    env.update(
        {
            "IMAGE": "registry.example.com/nvflare-parent:test",
            "REPO_ROOT": str(_REPO_ROOT),
            "WORK_DIR": str(tmp_path),
            **resources,
        }
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; init_k8s_env true; ensure_work_dirs; write_prepare_config nvflare-server nvflare-ws-server',
            "--",
            str(_K8S_COMMON),
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    return Path(result.stdout.strip()).read_text()


def test_prepare_config_uses_crc_sized_parent_requests_by_default(tmp_path):
    config = _write_prepare_config(tmp_path)

    assert 'cpu: "500m"' in config
    assert 'memory: "1Gi"' in config


def test_prepare_config_allows_parent_request_overrides(tmp_path):
    config = _write_prepare_config(tmp_path, PARENT_CPU="1", PARENT_MEMORY="4Gi")

    assert 'cpu: "1"' in config
    assert 'memory: "4Gi"' in config


def test_downloaded_image_builder_checks_out_provenance_revision(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "NVFlare Test")
    _git(repository, "config", "user.email", "nvflare-test@example.com")
    for relative in ("versioneer.py", "setup.cfg", "nvflare/_version.py"):
        target = repository / relative
        target.parent.mkdir(exist_ok=True)
        shutil.copy(_REPO_ROOT / relative, target)
    docker = repository / "docker"
    docker.mkdir()
    for name in ("Dockerfile.parent", "Dockerfile.job"):
        (docker / name).write_text("FROM scratch\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "test source")
    _git(repository, "tag", "2.10.0dev0")
    revision = _git(repository, "rev-parse", "HEAD")

    download_root = tmp_path / "downloaded"
    example = download_root / "examples" / "devops" / "openshift"
    shutil.copytree(_BUILD_IMAGES.parent.parent, example)
    (download_root / ".nvflare-example.json").write_text(
        json.dumps({"revision": revision, "nvflare_version": "2.10.0.dev1"}), encoding="utf-8"
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "nvflare").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
test "$1 $2 $3" = "examples revision --dir"
test "$4" = "$EXPECTED_DOWNLOAD_ROOT"
python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["revision"])' "$4/.nvflare-example.json"
""",
        encoding="utf-8",
    )
    container = binaries / "container-test"
    container.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == build ]]; then
  context="${!#}"
  test "$(git -C "$context" rev-parse HEAD)" = "$EXPECTED_REVISION"
  (cd "$context" && python3 -c "import versioneer; assert not versioneer.get_versions()['error']")
fi
printf '%s\n' "$1" >> "$CONTAINER_LOG"
""",
        encoding="utf-8",
    )
    for executable in (binaries / "nvflare", container):
        executable.chmod(0o755)
    command_log = tmp_path / "container.log"
    env = {
        **os.environ,
        "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
        "CONTAINER_TOOL": container.name,
        "CONTAINER_LOG": str(command_log),
        "EXPECTED_DOWNLOAD_ROOT": str(download_root),
        "EXPECTED_REVISION": revision,
        "NVFL_SOURCE_REPOSITORY": str(repository),
        "PARENT_IMAGE": "registry.example.com/nvflare-parent:test",
        "WORKLOAD_IMAGE": "registry.example.com/nvflare-job:test",
    }

    subprocess.run(["bash", str(example / "scripts" / "build_images.sh")], check=True, env=env)

    assert command_log.read_text(encoding="utf-8").splitlines() == ["build", "build", "push", "push"]

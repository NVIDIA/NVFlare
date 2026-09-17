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

import os
import shutil
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_K8S_COMMON = _REPO_ROOT / "examples" / "devops" / "openshift" / "scripts" / "k8s_common.sh"


def _write_prepare_config(tmp_path: Path, **resources: str) -> str:
    env = os.environ.copy()
    env.pop("PARENT_CPU", None)
    env.pop("PARENT_MEMORY", None)
    env.update(
        {
            "IMAGE": "registry.example.com/nvflare-parent:test",
            "EXAMPLE_ROOT": str(_REPO_ROOT / "examples" / "devops" / "openshift"),
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


def test_exported_numpy_job_uses_only_the_downloaded_openshift_example(tmp_path):
    example = tmp_path / "downloaded" / "examples" / "devops" / "openshift"
    shutil.copytree(_K8S_COMMON.parent.parent, example)
    work_dir = tmp_path / "work"
    env = {
        **os.environ,
        "EXAMPLE_ROOT": str(example),
        "WORK_DIR": str(work_dir),
        "CLIENTS": "site-1 site-2",
        "NUM_ROUNDS": "1",
    }

    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; init_k8s_env false; export_hello_numpy_job "$JOB_DIR"',
            "--",
            str(example / "scripts" / "k8s_common.sh"),
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    job = work_dir / "jobs" / "hello-numpy-k8s"
    assert (job / "meta.json").is_file()
    assert (job / "app" / "custom" / "numpy_client.py").read_bytes() == (
        example / "jobs" / "numpy_client.py"
    ).read_bytes()
    assert not (tmp_path / "downloaded" / "examples" / "hello-world").exists()

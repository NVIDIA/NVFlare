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
import subprocess
import sys

from nvflare.utils.tensor_disk_offload import (
    _OWNER_FILE,
    cleanup_owned_tensor_disk_offload_roots,
    create_tensor_disk_offload_root,
)


def test_surviving_parent_removes_only_exact_owned_job_roots(tmp_path):
    owned_a = create_tensor_disk_offload_root("job-a", temp_root=str(tmp_path))
    owned_b = create_tensor_disk_offload_root("job-a", temp_root=str(tmp_path))
    other_job = create_tensor_disk_offload_root("job-b", temp_root=str(tmp_path))
    with open(os.path.join(owned_a, "partial.safetensors"), "wb") as f:
        f.write(b"partial")

    cleanup = cleanup_owned_tensor_disk_offload_roots(
        job_id="job-a", owner_parent_pid=os.getppid(), temp_root=str(tmp_path)
    )

    assert sorted(cleanup.removed) == sorted([owned_a, owned_b])
    assert cleanup.failures == {}
    assert not os.path.exists(owned_a)
    assert not os.path.exists(owned_b)
    assert os.path.isdir(other_job)


def test_parent_reclaims_root_created_by_exited_child_process(tmp_path):
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from nvflare.utils.tensor_disk_offload import create_tensor_disk_offload_root; "
                "print(create_tensor_disk_offload_root('job-a', temp_root=sys.argv[1]), flush=True)"
            ),
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    child_root = child.stdout.strip()
    assert os.path.isdir(child_root)

    cleanup = cleanup_owned_tensor_disk_offload_roots(
        job_id="job-a", owner_parent_pid=os.getpid(), temp_root=str(tmp_path)
    )

    assert cleanup.removed == [child_root]
    assert cleanup.failures == {}
    assert not os.path.exists(child_root)


def test_cleanup_rejects_wrong_parent_and_replaced_root_identity(tmp_path):
    wrong_parent = create_tensor_disk_offload_root("job-a", temp_root=str(tmp_path))
    replaced = create_tensor_disk_offload_root("job-a", temp_root=str(tmp_path))
    marker_path = os.path.join(replaced, _OWNER_FILE)
    with open(marker_path, "r", encoding="utf-8") as marker:
        owner = json.load(marker)
    owner["root_inode"] += 1
    with open(marker_path, "w", encoding="utf-8") as marker:
        json.dump(owner, marker)

    cleanup = cleanup_owned_tensor_disk_offload_roots(
        job_id="job-a", owner_parent_pid=os.getppid() + 1, temp_root=str(tmp_path)
    )
    assert cleanup.removed == []

    cleanup = cleanup_owned_tensor_disk_offload_roots(
        job_id="job-a", owner_parent_pid=os.getppid(), temp_root=str(tmp_path)
    )
    assert cleanup.removed == [wrong_parent]
    assert os.path.isdir(replaced)


def test_cleanup_ignores_unowned_malformed_and_symlink_entries(tmp_path):
    unowned = tmp_path / "nvflare_tensor_offload_job-a_unowned"
    unowned.mkdir()
    malformed = tmp_path / "nvflare_tensor_offload_job-a_malformed"
    malformed.mkdir()
    (malformed / _OWNER_FILE).write_text("not-json", encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()
    symlink = tmp_path / "nvflare_tensor_offload_job-a_symlink"
    symlink.symlink_to(external, target_is_directory=True)

    cleanup = cleanup_owned_tensor_disk_offload_roots(
        job_id="job-a", owner_parent_pid=os.getppid(), temp_root=str(tmp_path)
    )

    assert cleanup.removed == []
    assert cleanup.failures == {}
    assert unowned.is_dir()
    assert malformed.is_dir()
    assert symlink.is_symlink()
    assert external.is_dir()

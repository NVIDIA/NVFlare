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

import pytest

import nvflare.utils.tensor_disk_offload as tensor_disk_offload
from nvflare.utils.tensor_disk_offload import (
    _CLEANUP_ROOT_PREFIX,
    _OWNER_FILE,
    cleanup_owned_tensor_disk_offload_roots,
    create_tensor_disk_offload_root,
)

requires_secure_cleanup = pytest.mark.skipif(
    not tensor_disk_offload._SECURE_CLEANUP_SUPPORTED,
    reason="platform does not support secure directory-fd cleanup",
)


@requires_secure_cleanup
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
    assert list(tmp_path.glob(f"{_CLEANUP_ROOT_PREFIX}*")) == []


@requires_secure_cleanup
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


@requires_secure_cleanup
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


@requires_secure_cleanup
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


@requires_secure_cleanup
def test_cleanup_preserves_path_replacement_raced_before_recursive_delete(tmp_path, monkeypatch):
    owned = create_tensor_disk_offload_root("job-a", temp_root=str(tmp_path))
    original = tmp_path / "original-owned-root"
    replacement_sentinel = "do-not-delete"
    real_rename = os.rename

    def replace_before_quarantine(src, dst, *args, **kwargs):
        if src == owned:
            real_rename(src, original)
            os.mkdir(src)
            with open(os.path.join(src, replacement_sentinel), "w", encoding="utf-8") as sentinel:
                sentinel.write("replacement")
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(tensor_disk_offload.os, "rename", replace_before_quarantine)

    cleanup = cleanup_owned_tensor_disk_offload_roots(
        job_id="job-a", owner_parent_pid=os.getppid(), temp_root=str(tmp_path)
    )

    assert cleanup.removed == []
    assert owned in cleanup.failures
    assert "identity changed while quarantining" in cleanup.failures[owned]
    assert original.is_dir()
    assert (tmp_path / os.path.basename(owned) / replacement_sentinel).read_text(encoding="utf-8") == "replacement"
    assert list(tmp_path.glob(f"{_CLEANUP_ROOT_PREFIX}*")) == []


@requires_secure_cleanup
def test_cleanup_rolls_back_partial_delete_for_a_later_retry(tmp_path, monkeypatch):
    owned = create_tensor_disk_offload_root("job-a", temp_root=str(tmp_path))
    data_path = os.path.join(owned, "partial.safetensors")
    with open(data_path, "wb") as data:
        data.write(b"partial")
    real_unlink = os.unlink
    failed_once = False

    def fail_data_unlink_once(path, *args, **kwargs):
        nonlocal failed_once
        if path == "partial.safetensors" and not failed_once:
            failed_once = True
            raise OSError("injected unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(tensor_disk_offload.os, "unlink", fail_data_unlink_once)

    first = cleanup_owned_tensor_disk_offload_roots(
        job_id="job-a", owner_parent_pid=os.getppid(), temp_root=str(tmp_path)
    )

    assert first.removed == []
    assert "injected unlink failure" in first.failures[owned]
    assert os.path.isdir(owned)
    assert os.path.isfile(os.path.join(owned, _OWNER_FILE))
    assert list(tmp_path.glob(f"{_CLEANUP_ROOT_PREFIX}*")) == []

    second = cleanup_owned_tensor_disk_offload_roots(
        job_id="job-a", owner_parent_pid=os.getppid(), temp_root=str(tmp_path)
    )
    assert second.removed == [owned]
    assert second.failures == {}
    assert not os.path.exists(owned)
    assert list(tmp_path.glob(f"{_CLEANUP_ROOT_PREFIX}*")) == []


@pytest.mark.parametrize("replacement_kind", ["file", "directory"])
@requires_secure_cleanup
def test_cleanup_rollback_preserves_competing_path(tmp_path, monkeypatch, replacement_kind):
    owned = create_tensor_disk_offload_root("job-a", temp_root=str(tmp_path))
    owned_stat = os.stat(owned, follow_symlinks=False)
    owned_name = os.path.basename(owned)

    def fail_after_creating_competing_path(*args, **kwargs):
        if replacement_kind == "file":
            with open(owned, "w", encoding="utf-8") as replacement:
                replacement.write("unrelated")
        else:
            os.mkdir(owned)
        raise OSError("injected recursive cleanup failure")

    monkeypatch.setattr(tensor_disk_offload, "_remove_tree_at", fail_after_creating_competing_path)

    cleanup = cleanup_owned_tensor_disk_offload_roots(
        job_id="job-a", owner_parent_pid=os.getppid(), temp_root=str(tmp_path)
    )

    assert cleanup.removed == []
    assert "rollback failed" in cleanup.failures[owned]
    if replacement_kind == "file":
        assert (tmp_path / owned_name).read_text(encoding="utf-8") == "unrelated"
    else:
        assert (tmp_path / owned_name).is_dir()
        assert list((tmp_path / owned_name).iterdir()) == []

    quarantine_dirs = list(tmp_path.glob(f"{_CLEANUP_ROOT_PREFIX}*"))
    assert len(quarantine_dirs) == 1
    quarantined_root = quarantine_dirs[0] / owned_name
    quarantined_stat = os.stat(quarantined_root, follow_symlinks=False)
    assert (quarantined_stat.st_dev, quarantined_stat.st_ino) == (owned_stat.st_dev, owned_stat.st_ino)


@requires_secure_cleanup
def test_cleanup_restores_owner_record_when_final_root_removal_fails(tmp_path, monkeypatch):
    owned = create_tensor_disk_offload_root("job-a", temp_root=str(tmp_path))
    owned_name = os.path.basename(owned)
    real_rmdir = os.rmdir
    failed_once = False

    def fail_final_root_rmdir_once(path, *args, **kwargs):
        nonlocal failed_once
        if path == owned_name and kwargs.get("dir_fd") is not None and not failed_once:
            failed_once = True
            raise OSError("injected final rmdir failure")
        return real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(tensor_disk_offload.os, "rmdir", fail_final_root_rmdir_once)

    first = cleanup_owned_tensor_disk_offload_roots(
        job_id="job-a", owner_parent_pid=os.getppid(), temp_root=str(tmp_path)
    )

    assert first.removed == []
    assert "injected final rmdir failure" in first.failures[owned]
    assert os.path.isfile(os.path.join(owned, _OWNER_FILE))
    assert list(tmp_path.glob(f"{_CLEANUP_ROOT_PREFIX}*")) == []

    second = cleanup_owned_tensor_disk_offload_roots(
        job_id="job-a", owner_parent_pid=os.getppid(), temp_root=str(tmp_path)
    )
    assert second.removed == [owned]
    assert second.failures == {}
    assert not os.path.exists(owned)


def test_tensor_offload_is_explicitly_gated_without_secure_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(tensor_disk_offload, "_SECURE_CLEANUP_SUPPORTED", False)

    with pytest.raises(RuntimeError, match="secure directory-fd cleanup"):
        create_tensor_disk_offload_root("job-a", temp_root=str(tmp_path))

    cleanup = cleanup_owned_tensor_disk_offload_roots("job-a", temp_root=str(tmp_path))
    assert cleanup.removed == []
    assert cleanup.failures == {str(tmp_path): tensor_disk_offload._UNSUPPORTED_CLEANUP_ERROR}

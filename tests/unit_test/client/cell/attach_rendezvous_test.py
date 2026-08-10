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

import errno
import json
import os
import stat
import threading

import pytest

from nvflare.client.cell import attach_rendezvous
from nvflare.client.cell.attach_rendezvous import (
    ATTACH_ENDPOINT_FILE,
    ATTACH_RENDEZVOUS_FILE_MODE,
    AttachEndpointKey,
    AttachEndpointOwnershipError,
    AttachEndpointPublisher,
    attach_claim_dir,
    wait_for_attach_endpoint,
)


def _make_listener(root_dir, name="lst_12345678"):
    listener = root_dir / name
    listener.mkdir(mode=0o770)
    (listener / "conns").mkdir(mode=0o770)
    marker = listener / ".nvf_file_transport"
    marker.touch(mode=0o660)
    os.chmod(marker, 0o660)
    return f"shared-file://0{listener}"


def _publish(root_dir, connect_url=None):
    connect_url = connect_url or _make_listener(root_dir)
    publisher = AttachEndpointPublisher(str(root_dir), "site-1", "trainer_a")
    publisher.publish(
        cj_fqcn="site-1.job-1",
        trainer_fqcn="site-1.-client_api_trainer_a",
        connect_url=connect_url,
        connection_security="clear",
    )
    return publisher


def test_publish_wait_and_close_round_trip(tmp_path):
    publisher = _publish(tmp_path)

    record = wait_for_attach_endpoint(str(tmp_path), "site-1", "trainer_a", timeout=0.5)

    assert record[AttachEndpointKey.CJ_FQCN] == "site-1.job-1"
    assert record[AttachEndpointKey.TRAINER_FQCN] == "site-1.-client_api_trainer_a"
    claim_dir = attach_claim_dir(str(tmp_path), "site-1", "trainer_a")
    assert os.stat(os.path.join(claim_dir, ATTACH_ENDPOINT_FILE)).st_mode & 0o777 == 0o660

    publisher.close()
    assert not os.path.exists(claim_dir)


def test_claim_lock_keeps_shared_group_access_under_restrictive_umask(tmp_path):
    root_dir = tmp_path / "shared"
    root_dir.mkdir(mode=0o770)
    os.chmod(root_dir, 0o770)
    previous_umask = os.umask(0o027)
    try:
        publisher = _publish(root_dir)
    finally:
        os.umask(previous_umask)

    try:
        claim_dir = attach_claim_dir(str(root_dir), "site-1", "trainer_a")
        lock_path = attach_rendezvous._claim_lock_path(os.path.dirname(claim_dir), "trainer_a")
        assert stat.S_IMODE(os.stat(lock_path).st_mode) == 0o660
        assert wait_for_attach_endpoint(str(root_dir), "site-1", "trainer_a", timeout=0.5)
    finally:
        publisher.close()


def test_live_attach_id_claim_cannot_be_stolen(tmp_path):
    publisher = _publish(tmp_path)
    try:
        with pytest.raises(AttachEndpointOwnershipError, match="already claimed"):
            AttachEndpointPublisher(str(tmp_path), "site-1", "trainer_a")
    finally:
        publisher.close()


def test_orphaned_claim_is_recovered_without_old_owner_deleting_new_claim(tmp_path):
    old = _publish(tmp_path)
    attach_rendezvous._release_claim_lock(old._claim_lock_fd)
    old._claim_lock_fd = None
    old._closed = True

    new = _publish(tmp_path, connect_url=f"shared-file://0{tmp_path}/lst_12345678")
    record = wait_for_attach_endpoint(str(tmp_path), "site-1", "trainer_a", timeout=0.5)

    assert record[AttachEndpointKey.INSTANCE_ID] == new.instance_id
    new.close()


def test_wait_rejects_endpoint_after_publisher_lock_is_released(tmp_path, monkeypatch):
    publisher = _publish(tmp_path)
    attach_rendezvous._release_claim_lock(publisher._claim_lock_fd)
    publisher._claim_lock_fd = None
    monkeypatch.setattr(attach_rendezvous, "_WAIT_INTERVAL", 0.001)

    try:
        with pytest.raises(TimeoutError, match="no live attach endpoint"):
            wait_for_attach_endpoint(str(tmp_path), "site-1", "trainer_a", timeout=0.02)
    finally:
        publisher.close()


def test_wait_does_not_use_cross_node_wall_clock(tmp_path, monkeypatch):
    publisher = _publish(tmp_path)
    monkeypatch.setattr(attach_rendezvous.time, "time", lambda: (_ for _ in ()).throw(AssertionError("wall clock")))
    try:
        record = wait_for_attach_endpoint(str(tmp_path), "site-1", "trainer_a", timeout=0.5)
        assert record[AttachEndpointKey.INSTANCE_ID] == publisher.instance_id
    finally:
        publisher.close()


def test_wait_retries_transient_filesystem_read_error(tmp_path, monkeypatch):
    publisher = _publish(tmp_path)
    read_valid_endpoint = attach_rendezvous._read_valid_endpoint
    attempts = 0

    def transient_then_read(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(errno.ESTALE, "stale file handle")
        return read_valid_endpoint(*args)

    monkeypatch.setattr(attach_rendezvous, "_read_valid_endpoint", transient_then_read)
    monkeypatch.setattr(attach_rendezvous, "_WAIT_INTERVAL", 0.001)
    try:
        record = wait_for_attach_endpoint(str(tmp_path), "site-1", "trainer_a", timeout=0.5)
        assert record[AttachEndpointKey.INSTANCE_ID] == publisher.instance_id
        assert attempts == 2
    finally:
        publisher.close()


def test_reader_rejects_world_writable_trust_root(tmp_path):
    publisher = _publish(tmp_path)
    os.chmod(tmp_path, 0o777)
    try:
        with pytest.raises(RuntimeError, match="world-writable"):
            wait_for_attach_endpoint(str(tmp_path), "site-1", "trainer_a", timeout=0.5)
    finally:
        os.chmod(tmp_path, 0o700)
        publisher.close()


def test_reader_rejects_listener_outside_configured_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir(mode=0o770)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o770)
    outside_url = _make_listener(outside)
    publisher = _publish(root)
    claim_dir = attach_claim_dir(str(root), "site-1", "trainer_a")
    endpoint_path = os.path.join(claim_dir, ATTACH_ENDPOINT_FILE)
    with open(endpoint_path, "r", encoding="utf-8") as f:
        record = json.load(f)
    record[AttachEndpointKey.CONNECT_URL] = outside_url
    attach_rendezvous._atomic_write_json(endpoint_path, record, ATTACH_RENDEZVOUS_FILE_MODE)

    try:
        with pytest.raises(RuntimeError, match="immediate child"):
            wait_for_attach_endpoint(str(root), "site-1", "trainer_a", timeout=0.5)
    finally:
        publisher.close()


def test_record_cannot_extend_liveness_after_publisher_lock_is_released(tmp_path, monkeypatch):
    publisher = _publish(tmp_path)
    claim_dir = attach_claim_dir(str(tmp_path), "site-1", "trainer_a")
    endpoint_path = os.path.join(claim_dir, ATTACH_ENDPOINT_FILE)
    with open(endpoint_path, "r", encoding="utf-8") as f:
        record = json.load(f)
    record["lease_timeout"] = 3600
    attach_rendezvous._atomic_write_json(endpoint_path, record, ATTACH_RENDEZVOUS_FILE_MODE)
    attach_rendezvous._release_claim_lock(publisher._claim_lock_fd)
    publisher._claim_lock_fd = None
    monkeypatch.setattr(attach_rendezvous, "_WAIT_INTERVAL", 0.001)

    try:
        with pytest.raises(TimeoutError, match="no live attach endpoint"):
            wait_for_attach_endpoint(str(tmp_path), "site-1", "trainer_a", timeout=0.02)
    finally:
        publisher.close()


def test_wait_can_be_cancelled_without_a_timeout(tmp_path):
    stopped = threading.Event()
    stopped.set()

    with pytest.raises(attach_rendezvous.AttachRendezvousCancelled, match="stopped"):
        wait_for_attach_endpoint(str(tmp_path), "site-1", "trainer_a", timeout=None, stop_event=stopped)

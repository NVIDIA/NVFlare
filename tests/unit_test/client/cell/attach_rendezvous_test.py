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
import time

import pytest

from nvflare.client.cell import attach_rendezvous
from nvflare.client.cell.attach_rendezvous import (
    ATTACH_ENDPOINT_FILE,
    ATTACH_LEASE_FILE,
    ATTACH_RENDEZVOUS_LEASE_TIMEOUT,
    AttachEndpointKey,
    AttachEndpointPublisher,
    attach_claim_dir,
    wait_for_attach_endpoint,
)


def _publish(root_dir):
    publisher = AttachEndpointPublisher(str(root_dir), "site-1", "trainer_a")
    publisher.publish(
        cj_fqcn="site-1.job-1",
        trainer_fqcn="site-1.job-1.-client_api_trainer_a",
        connect_url=f"shared-file://0{root_dir}/lst_12345678",
        connection_security="clear",
    )
    return publisher


def test_publish_wait_and_close_round_trip(tmp_path):
    publisher = _publish(tmp_path)

    record = wait_for_attach_endpoint(str(tmp_path), "site-1", "trainer_a", timeout=0.1)

    assert record[AttachEndpointKey.CJ_FQCN] == "site-1.job-1"
    assert record[AttachEndpointKey.TRAINER_FQCN] == "site-1.job-1.-client_api_trainer_a"
    claim_dir = attach_claim_dir(str(tmp_path), "site-1", "trainer_a")
    assert os.stat(os.path.join(claim_dir, ATTACH_ENDPOINT_FILE)).st_mode & 0o777 == 0o660

    publisher.close()
    assert not os.path.exists(claim_dir)


def test_live_attach_id_claim_cannot_be_stolen(tmp_path):
    publisher = _publish(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already claimed"):
            AttachEndpointPublisher(str(tmp_path), "site-1", "trainer_a")
    finally:
        publisher.close()


def test_stale_claim_is_recovered_without_old_owner_deleting_new_claim(tmp_path):
    old = _publish(tmp_path)
    claim_dir = attach_claim_dir(str(tmp_path), "site-1", "trainer_a")
    stale_time = time.time() - ATTACH_RENDEZVOUS_LEASE_TIMEOUT - 1
    os.utime(os.path.join(claim_dir, ATTACH_LEASE_FILE), (stale_time, stale_time))

    new = _publish(tmp_path)
    old.close()

    record = wait_for_attach_endpoint(str(tmp_path), "site-1", "trainer_a", timeout=0.1)
    assert record[AttachEndpointKey.INSTANCE_ID] == new.instance_id
    new.close()


def test_wait_ignores_stale_record_and_times_out(tmp_path, monkeypatch):
    publisher = _publish(tmp_path)
    claim_dir = attach_claim_dir(str(tmp_path), "site-1", "trainer_a")
    stale_time = time.time() - ATTACH_RENDEZVOUS_LEASE_TIMEOUT - 1
    os.utime(os.path.join(claim_dir, ATTACH_LEASE_FILE), (stale_time, stale_time))
    monkeypatch.setattr(attach_rendezvous, "_WAIT_INTERVAL", 0.001)

    try:
        with pytest.raises(TimeoutError, match="no live attach endpoint"):
            wait_for_attach_endpoint(str(tmp_path), "site-1", "trainer_a", timeout=0.01)
    finally:
        publisher.close()

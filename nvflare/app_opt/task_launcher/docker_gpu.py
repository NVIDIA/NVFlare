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

"""Docker-native ownership records for an administrator-approved GPU partition.

Each reservation is a never-started container with a daemon-wide device name.
No guardian process runs. All users of the approved partition must participate
in this admission protocol, on one daemon; other host schedulers must exclude
the partition. Visibility alone does not establish exclusive ownership.
"""

import hashlib
import json
import uuid

from nvflare.apis.task_launcher_spec import TaskLauncherError
from nvflare.app_opt.job_launcher import docker_launcher as backend

GPU_OWNER = "nvflare.gpu.owner"
GPU_DEVICE = "nvflare.gpu.device"
GPU_PARTITION = "nvflare.gpu.partition"


def validate_partition(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"id", "devices", "exclusive"}:
        raise ValueError("gpu_partition requires id, devices and exclusive")
    if value["exclusive"] is not True:
        raise ValueError("gpu_partition must be approved exclusive to participating Docker launchers")
    if not isinstance(value["id"], str) or not value["id"].strip() or "\x00" in value["id"]:
        raise ValueError("gpu_partition id must be nonempty text without NUL")
    devices = value["devices"]
    if not isinstance(devices, list) or not devices:
        raise ValueError("gpu_partition devices must be a nonempty list of full GPU UUIDs")
    normalized = []
    for device in devices:
        if not isinstance(device, str) or not device.startswith("GPU-"):
            raise ValueError("gpu_partition devices must be full physical GPU UUIDs")
        try:
            canonical = "GPU-" + str(uuid.UUID(device[4:]))
        except ValueError as error:
            raise ValueError("gpu_partition devices must be full physical GPU UUIDs") from error
        if canonical.lower() != device.lower():
            raise ValueError("gpu_partition devices must use the full hyphenated GPU UUID")
        normalized.append(canonical)
    if len(set(normalized)) != len(normalized):
        raise ValueError("gpu_partition devices must be distinct")
    return {"id": value["id"], "devices": tuple(sorted(normalized)), "exclusive": True}


def validate_resident_access(partition, container_kwargs):
    """An opted-in site's resident launches cannot bypass its task partition."""
    if any(container_kwargs.get(name) for name in ("privileged", "devices", "device_cgroup_rules", "volumes_from")):
        raise ValueError("resident Docker access cannot bypass the approved task GPU partition")
    devices = set()
    for request in container_kwargs.get("device_requests", []):
        ids = request.get("DeviceIDs", [])
        if not ids or request.get("Count", 0):
            raise ValueError("resident Docker GPU access requires explicit UUIDs outside the task GPU partition")
        for device in ids:
            normalized = validate_partition({"id": "resident", "devices": [device], "exclusive": True})["devices"][0]
            if normalized in partition["devices"]:
                raise ValueError("resident Docker job requests a GPU reserved for task admission")
            devices.add(normalized)
    return tuple(sorted(devices))


class _DeviceReservation:
    def __init__(self, client, device, owner, partition_id):
        self.client, self.device, self.owner, self.partition_id = client, device, owner, partition_id
        self.name = "nvflare-gpu-" + hashlib.sha256(device.encode()).hexdigest()
        self.container_id = None
        self.removal_requested = False
        self.released = False

    def inspect(self):
        container = self.client.containers.get(self.container_id or self.name)
        labels = container.attrs.get("Config", {}).get("Labels", {})
        if labels.get(GPU_OWNER) != self.owner or labels.get(GPU_DEVICE) != self.device:
            raise TaskLauncherError("GPU reservation ownership changed; refusing container control")
        if self.container_id is not None and container.id != self.container_id:
            raise TaskLauncherError("GPU reservation physical identity changed")
        if container.status != backend.DockerStatus.CREATED:
            raise TaskLauncherError("GPU reservation record was unexpectedly started")
        self.container_id = container.id
        return container

    def release(self):
        if self.released:
            return
        try:
            container = self.inspect()
            self.removal_requested = True
            container.remove(force=True, v=True)
        except backend.docker.errors.NotFound:
            if not self.removal_requested:
                raise TaskLauncherError("GPU reservation disappeared before owned release")
        except Exception:
            # A lost remove response may still have completed. Its subsequent
            # absence is proof only after we issued removal of this owned ID.
            if not self.removal_requested:
                raise
        try:
            self.inspect()
        except backend.docker.errors.NotFound:
            if self.removal_requested:
                self.released = True
                return
        raise TaskLauncherError("GPU reservation removal is unconfirmed")


class DockerGpuReservations:
    """Retain every acquired/uncertain reservation until explicit safe release."""

    def __init__(self, client, partition, image, uid_gid, owner, identity=()):
        self.client, self.partition, self.image, self.uid_gid, self.owner = client, partition, image, uid_gid, owner
        self.reservations = []
        self.identity = tuple(identity)

    @property
    def devices(self):
        return tuple(reservation.device for reservation in self.reservations if not reservation.released)

    def acquire(self, count):
        if not count:
            return
        if self.partition is None or count > len(self.partition["devices"]):
            raise TaskLauncherError("GPU request exceeds the approved Docker partition")
        for device in self.partition["devices"]:
            reservation = _DeviceReservation(self.client, device, self.owner, self.partition["id"])
            # Retain identity before the mutating API: an unacknowledged create
            # must not be mistaken for an unreserved device.
            self.reservations.append(reservation)
            try:
                container = self.client.containers.create(
                    self.image,
                    name=reservation.name,
                    entrypoint=[],
                    command=["/bin/false"],
                    labels={
                        GPU_OWNER: self.owner,
                        GPU_DEVICE: device,
                        GPU_PARTITION: self.partition["id"],
                        "nvflare.task.identity": json.dumps(self.identity),
                        "nvflare.task.resource": "gpu-reservation",
                    },
                    user=self.uid_gid,
                    network_mode="none",
                    cap_drop=["ALL"],
                    security_opt=["no-new-privileges:true"],
                    environment={"NVIDIA_VISIBLE_DEVICES": "void"},
                    auto_remove=False,
                    detach=True,
                )
                reservation.container_id = container.id
            except Exception as error:
                status = getattr(error, "status_code", None)
                if status == 409:
                    # The name already belongs to another admitted execution.
                    # This failed create acquired no resource; never remove it.
                    self.reservations.remove(reservation)
                    continue
                if isinstance(status, int) and 400 <= status < 500:
                    self.reservations.remove(reservation)
                    raise TaskLauncherError("Docker rejected GPU reservation creation") from error
                # For an ambiguous response, adopt only our exact label/UUID.
                # A missing record might still materialize later: retain it.
                reservation.inspect()
            reservation.inspect()
            if len(self.reservations) == count:
                return
        raise TaskLauncherError("approved Docker GPU partition has insufficient free capacity")

    def verify(self):
        for reservation in self.reservations:
            if not reservation.released:
                reservation.inspect()

    def release(self):
        failures = []
        for reservation in self.reservations:
            try:
                reservation.release()
            except Exception as error:
                failures.append(error)
        if failures:
            raise TaskLauncherError("Docker GPU reservation release is unconfirmed") from failures[0]

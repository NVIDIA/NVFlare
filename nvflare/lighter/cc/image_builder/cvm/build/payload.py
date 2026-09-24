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

"""Explicit Python payloads for measured guests and standalone host deliveries."""

import shutil
from pathlib import Path

# Keep deployment contents independent of the full build/admin source tree.
GUEST_MODULES = (
    "common.firewall",
    "common.contracts",
    "common.errors",
    "common.io",
    "common.linux",
    "common.luks",
    "common.measurements",
    "common.evidence",
    "common.gpu_claims",
    "common.validation",
    "common.services",
    "runtime.bootstrap",
    "runtime.supervisor",
    "runtime.systemd",
    "runtime.attestation",
    "runtime.integrity",
    "runtime.gpu",
    "runtime.gpu_claims",
    "runtime.audit",
    "runtime.platforms",
    "runtime.storage",
)
HOST_MODULES = (
    "common.contracts",
    "common.errors",
    "common.io",
    "common.linux",
    "common.measurements",
    "common.policy",
    "common.gpu_policy",
    "common.gpu_claims",
    "common.references",
    "artifacts.bundle",
    "host.launcher",
    "host.platforms",
)
# Preserve provenance of all image construction and deployment inputs, including
# the provisioner and these payload lists. Do not hash only the installed guest.
SOURCE_DIRECTORIES = ("cvm", "services", "initramfs", "templates")


def copy_modules(source, destination, modules):
    """Copy a closed module set and its package initializers, never caches."""
    source, destination = Path(source), Path(destination)
    files = {"cvm/__init__.py"}
    for module in modules:
        path = Path("cvm", *module.split(".")).with_suffix(".py")
        files.add(path.as_posix())
        for parent in path.parents:
            if parent == Path("."):
                break
            files.add((parent / "__init__.py").as_posix())
    for name in sorted(files):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)

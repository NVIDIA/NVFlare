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

"""Provision and harden the disposable Ubuntu construction guest."""

import argparse
import contextlib
import hashlib
import json
import os
import platform as system_platform
import re
import shutil
import stat
import subprocess
from pathlib import Path

DIRECTORIES = (
    "/usr/lib/cvm/bin",
    "/usr/lib/cvm/builder",
    "/etc/cvm",
    "/vault",
    "/applog",
    "/user_config",
    "/user_data",
    "/rofs",
    "/cow",
    "/etc/docker",
    "/etc/containerd",
    "/etc/systemd/system/docker.service.d",
    "/etc/systemd/system/docker.socket.d",
    "/etc/systemd/system/containerd.service.d",
)
BOOTSTRAP_UNITS = (
    "chrony.service",
    "cvm_time_sync.service",
    "cvm_vault.service",
    "cvm_firewall.service",
    "cvm_reference.service",
)
UPDATE_UNITS = (
    "apt-daily.service",
    "apt-daily.timer",
    "apt-daily-upgrade.service",
    "apt-daily-upgrade.timer",
    "unattended-upgrades.service",
    "snapd.service",
    "snapd.socket",
    "snapd.seeded.service",
    "update-notifier-download.service",
    "update-notifier-download.timer",
    "update-notifier-motd.service",
    "update-notifier-motd.timer",
)
CRASH_UNITS = (
    "apport.service",
    "apport-autoreport.service",
    "apport-autoreport.path",
    "apport-autoreport.timer",
    "systemd-coredump.socket",
    "systemd-coredump@.service",
)
LOGIN_UNITS = ("serial-getty@ttyS0.service", "getty@tty1.service", "emergency.service", "rescue.service")
DEV_DEPENDENCY_FILES = (
    "/usr/lib/systemd/system/cvm_workload.target",
    "/usr/lib/systemd/system/cvm_app.service",
    "/usr/lib/systemd/system/mount_user_data.service",
    "/usr/lib/systemd/system/periodic_attestation.service",
    "/usr/lib/systemd/system/periodic_attestation.timer",
    "/etc/systemd/system/docker.service.d/vault.conf",
    "/etc/systemd/system/docker.socket.d/vault.conf",
    "/etc/systemd/system/containerd.service.d/vault.conf",
)


def execute(argv, *, env=None):
    subprocess.run(argv, check=True, env=env)


def target(root, path):
    return Path(root) / Path(path).relative_to("/")


def write_file(root, path, content, mode=0o644):
    destination = target(root, path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content if isinstance(content, bytes) else content.encode())
    destination.chmod(mode)


def copy_file(source, root, path, mode):
    destination = target(root, path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(mode)


def load_config(path):
    value = json.loads(Path(path).read_text())
    required = {
        "build_id",
        "build_user",
        "dev_mode",
        "gpu",
        "guest_release",
        "kernel_version",
        "platform",
        "profile_version",
        "required_system_packages",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Invalid provisioning configuration fields")
    if not isinstance(value["build_id"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value["build_id"]):
        raise ValueError("Invalid provisioning identity")
    if not isinstance(value["build_user"], str) or not re.fullmatch(r"[a-z_][a-z0-9_-]*", value["build_user"]):
        raise ValueError("Invalid build user")
    if not isinstance(value["profile_version"], str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9_.-]{0,63}", value["profile_version"]
    ):
        raise ValueError("Invalid profile version")
    if value["guest_release"] != "26.04" or value["platform"] not in ("amd_sev_snp", "intel_tdx"):
        raise ValueError("Unsupported construction target")
    if value["gpu"] not in ("none", "nvidia_cc") or type(value["dev_mode"]) is not bool:
        raise ValueError("Invalid construction mode")
    if not isinstance(value["kernel_version"], str) or not re.fullmatch(r"[A-Za-z0-9.+:~_-]+", value["kernel_version"]):
        raise ValueError("Invalid kernel version")
    packages = value["required_system_packages"]
    package_pattern = re.compile(r"[a-z0-9][a-z0-9.+-]*(?::[a-z0-9-]+)?=[A-Za-z0-9][A-Za-z0-9.+:~_-]*")
    if (
        not isinstance(packages, list)
        or not packages
        or not all(isinstance(item, str) and package_pattern.fullmatch(item) for item in packages)
    ):
        raise ValueError("Invalid package pins")
    names = [item.split("=", 1)[0] for item in packages]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate package pin")
    return value


def docker_configuration(gpu):
    value = {
        "data-root": "/vault/docker/data",
        "exec-root": "/run/docker",
        "log-driver": "local",
        "storage-driver": "overlay2",
        "features": {"containerd-snapshotter": False},
    }
    if gpu == "nvidia_cc":
        value["runtimes"] = {"nvidia": {"path": "nvidia-container-runtime", "runtimeArgs": []}}
    return json.dumps(value, separators=(",", ":")) + "\n"


def install_files(config, payload, root=Path("/")):
    payload = Path(payload)
    root = Path(root)
    for path in DIRECTORIES:
        target(root, path).mkdir(parents=True, exist_ok=True, mode=0o755)

    for source in sorted((payload / "source/builder").glob("*.py")):
        copy_file(source, root, "/usr/lib/cvm/builder/" + source.name, 0o644)
    copy_file(payload / "inputs/kbs-client", root, "/usr/lib/cvm/bin/kbs-client", 0o755)
    for name in ("kbs-ca.pem", "as-public.pem", "runtime.json"):
        copy_file(payload / "inputs" / name, root, "/etc/cvm/" + name, 0o644)
    write_file(root, "/etc/cvm_build_id", config["build_id"] + "\n")
    write_file(root, "/etc/cvm_profile_version", config["profile_version"] + "\n")

    if config["gpu"] == "nvidia_cc":
        copy_file(payload / "inputs/gpu-policy.json", root, "/etc/cvm/gpu-policy.json", 0o644)
        library = target(root, "/usr/lib/x86_64-linux-gnu/libnvat.so.1.2.2")
        copy_file(payload / "inputs/libnvat.so.1.2.2", root, "/usr/lib/x86_64-linux-gnu/libnvat.so.1.2.2", 0o644)
        target(root, "/usr/lib/x86_64-linux-gnu/libnvat.so.1").symlink_to(library.name)
        target(root, "/usr/lib/x86_64-linux-gnu/libnvat.so").symlink_to(library.name)

    services = payload / "source/services"
    units = sorted(services.glob("*.service")) + sorted(services.glob("*.target")) + sorted(services.glob("*.timer"))
    if not units:
        raise ValueError("No measured systemd units")
    for source in units:
        copy_file(source, root, "/usr/lib/systemd/system/" + source.name, 0o644)
    for unit, name in (
        ("docker.service", "docker_vault.conf"),
        ("docker.socket", "socket_vault.conf"),
        ("containerd.service", "docker_vault.conf"),
    ):
        copy_file(services / name, root, f"/etc/systemd/system/{unit}.d/vault.conf", 0o644)

    write_file(root, "/etc/docker/daemon.json", docker_configuration(config["gpu"]))
    write_file(
        root, "/etc/containerd/config.toml", 'version = 2\nroot = "/vault/containerd"\nstate = "/run/containerd"\n'
    )
    for path in ("hooks/cvm_verity", "scripts/local-top/verity_root", "scripts/local-bottom/overlay_root"):
        copy_file(payload / "source/initramfs" / path, root, "/etc/initramfs-tools/" + path, 0o755)

    if config["dev_mode"]:
        write_file(root, "/etc/cvm/dev_mode", "Development only: no TEE and no KBS authorization.")
        for path in DEV_DEPENDENCY_FILES:
            location = target(root, path)
            location.write_text(location.read_text().replace("cvm_integrity.service", ""))

    module = "sev_guest" if config["platform"] == "amd_sev_snp" else "tdx_guest"
    write_file(root, "/etc/modules-load.d/cvm.conf", f"{module}\ndm_crypt\ndm_integrity\n")
    write_file(
        root,
        "/etc/fstab",
        "tmpfs /tmp tmpfs defaults,nosuid,nodev,mode=1777 0 0\n",
    )


@contextlib.contextmanager
def suppress_service_starts(path=Path("/usr/sbin/policy-rc.d")):
    path = Path(path)
    previous = None
    if os.path.lexists(path):
        if path.is_symlink():
            previous = ("symlink", os.readlink(path))
        elif path.is_file():
            previous = ("file", path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
        else:
            raise ValueError("Unsupported existing policy-rc.d")
        path.unlink()
    path.write_text("#!/bin/sh\nexit 101\n")
    path.chmod(0o755)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
        if previous and previous[0] == "symlink":
            path.symlink_to(previous[1])
        elif previous:
            path.write_bytes(previous[1])
            path.chmod(previous[2])


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_guest(config):
    release = system_platform.freedesktop_os_release()
    if release.get("ID") != "ubuntu" or release.get("VERSION_ID") != config["guest_release"]:
        raise ValueError("Construction guest release differs from the profile")
    if system_platform.machine() != "x86_64":
        raise ValueError("Construction guest must be x86-64")


def install(config, payload):
    verify_guest(config)
    environment = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
    with suppress_service_starts():
        execute(["apt-get", "update"], env=environment)
        execute(
            ["apt-get", "-y", "--no-install-recommends", "install", *config["required_system_packages"]],
            env=environment,
        )
    install_files(config, payload)
    if config["gpu"] == "nvidia_cc":
        execute(["ldconfig"])
        execute(["nvidia-ctk", "runtime", "configure", "--runtime=docker"])
        write_file(Path("/"), "/etc/docker/daemon.json", docker_configuration(config["gpu"]))
    execute(["swapoff", "-a"])
    execute(["systemctl", "daemon-reload"])
    execute(["update-initramfs", "-u", "-k", config["kernel_version"]])

    output = Path(payload) / "out"
    output.mkdir(mode=0o755)
    for source_name, destination_name in (("vmlinuz", "vmlinuz"), ("initrd.img", "initrd.img")):
        source = Path("/boot") / f"{source_name}-{config['kernel_version']}"
        destination = output / destination_name
        shutil.copyfile(source, destination)
        destination.chmod(0o644)
    manifest = {name: digest(output / name) for name in ("vmlinuz", "initrd.img")}
    (output / "artifacts.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    (output / "artifacts.json").chmod(0o644)


def mask(root, units):
    for unit in units:
        destination = target(root, "/etc/systemd/system/" + unit)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.is_dir() and not destination.is_symlink():
            raise ValueError(f"Cannot mask directory at {destination}")
        destination.unlink(missing_ok=True)
        destination.symlink_to("/dev/null")


def remove(path):
    path = Path(path)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def finalize(config, payload):
    # Each delivered guest negotiates its own time-service state. Build-time
    # NTS cookies must not hide missing key-exchange connectivity at first boot.
    execute(["systemctl", "stop", "chrony.service"])
    for path in Path("/var/lib/chrony").glob("*"):
        remove(path)
    execute(["systemctl", "disable", "ssh.service", "ssh.socket"])
    execute(["systemctl", "disable", "docker.service", "docker.socket", "containerd.service"])
    mask(Path("/"), ("ssh.service", "ssh.socket", *LOGIN_UNITS, *UPDATE_UNITS, *CRASH_UNITS))
    write_file(Path("/"), "/etc/cloud/cloud-init.disabled", "")
    write_file(
        Path("/"),
        "/etc/sysctl.d/99-cvm-no-coredumps.conf",
        "kernel.core_pattern = /dev/null\nkernel.core_uses_pid = 0\nfs.suid_dumpable = 0\n",
    )
    execute(["systemctl", "daemon-reload"])
    execute(["systemctl", "enable", *BOOTSTRAP_UNITS])
    execute(["passwd", "-l", "root"])
    execute(["passwd", "-l", config["build_user"]])
    for path in (
        "/root/.ssh",
        f"/home/{config['build_user']}/.ssh",
        "/var/lib/cloud",
        "/var/lib/docker",
        "/var/lib/containerd",
        "/var/log/cloud-init.log",
        "/var/log/cloud-init-output.log",
    ):
        remove(path)
    remove(payload)
    execute(["sync"])
    execute(["systemd-run", "--unit=cvm-construction-poweroff", "--on-active=5s", "/usr/bin/systemctl", "poweroff"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("install", "finalize"))
    parser.add_argument("config")
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("construction provisioning must run as root")
    configuration = load_config(args.config)
    payload = Path(args.config).resolve().parent
    if args.phase == "install":
        install(configuration, payload)
    else:
        finalize(configuration, payload)


if __name__ == "__main__":
    main()

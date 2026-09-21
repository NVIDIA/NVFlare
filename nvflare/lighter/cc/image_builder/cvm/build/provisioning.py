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
from urllib.parse import urlparse

DIRECTORIES = (
    "/usr/lib/cvm/bin",
    "/usr/lib/cvm/cvm",
    "/etc/cvm",
    "/vault",
    "/applog",
    "/user_config",
    "/user_data",
    "/rofs",
    "/cow",
    "/etc/docker",
    "/etc/containerd",
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


# Kernel settings applied by systemd-sysctl at every boot of the measured root.
# Root in the guest must not be able to load an unmeasured kernel, read kernel
# memory or dump kernel state to the host-visible console. The command line
# additionally enables lockdown and module signature enforcement.
HARDENING_SYSCTL = """# CVM guest hardening; the measured command line sets lockdown and loglevel.
kernel.core_pattern = /dev/null
kernel.core_uses_pid = 0
fs.suid_dumpable = 0
kernel.kexec_load_disabled = 1
kernel.sysrq = 0
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2
kernel.printk = 3 4 1 3
kernel.unprivileged_bpf_disabled = 1
kernel.yama.ptrace_scope = 1
kernel.perf_event_paranoid = 3
dev.tty.ldisc_autoload = 0
fs.protected_symlinks = 1
fs.protected_hardlinks = 1
fs.protected_fifos = 2
fs.protected_regular = 2
vm.unprivileged_userfaultfd = 0
"""


# Construction artifacts that would otherwise give every guest booted from the
# bundle the same identity, or leak build-host history into the public root.
IDENTITY_FILES = (
    "/var/lib/systemd/random-seed",
    "/var/log/dpkg.log",
    "/var/log/alternatives.log",
    "/var/log/cloud-init.log",
    "/var/log/cloud-init-output.log",
    "/root/.bash_history",
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


def validate_time_servers(servers):
    """Accept a non-empty list of NTS-capable time server host names or addresses."""
    if not isinstance(servers, list) or not servers:
        raise ValueError("time_servers must be a non-empty list of NTS server names")
    for server in servers:
        if not isinstance(server, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]{0,252}", server):
            raise ValueError("Invalid time server name")
    if len(set(servers)) != len(servers):
        raise ValueError("Duplicate time server")
    return servers


def chrony_configuration(servers):
    """Render a chrony configuration that trusts only authenticated NTS sources.

    The distribution pools and DHCP-supplied servers are omitted on purpose: the
    untrusted host controls DHCP and the network, so an unauthenticated source
    would let it steer the guest clock that gates attestation freshness.
    """
    lines = ["# Measured CVM time sources: NTS-authenticated only; no pools, no DHCP servers."]
    lines += [f"server {server} iburst nts" for server in validate_time_servers(servers)]
    lines += [
        "authselectmode require",
        "driftfile /var/lib/chrony/chrony.drift",
        "ntsdumpdir /var/lib/chrony",
        "makestep 1 3",
        "maxupdateskew 100.0",
        "rtcsync",
        "leapsectz right/UTC",
    ]
    return "\n".join(lines) + "\n"


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
    if not isinstance(value, dict) or set(value) - {"apt_repositories", "time_servers"} != required:
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
    validate_apt_repositories(value.get("apt_repositories", []))
    if value["gpu"] == "nvidia_cc" and not value.get("apt_repositories"):
        raise ValueError("GPU construction requires authenticated package repositories")
    if "time_servers" in value:
        validate_time_servers(value["time_servers"])
    return value


def validate_apt_repositories(repositories):
    if not isinstance(repositories, list):
        raise ValueError("Invalid GPU apt repositories")
    for repository in repositories:
        if not isinstance(repository, dict) or set(repository) != {"url", "suite", "components", "keyring_sha256"}:
            raise ValueError("Invalid GPU apt repository fields")
        url = repository["url"]
        if not isinstance(url, str) or not re.fullmatch(r"https://[A-Za-z0-9._:/-]+", url):
            raise ValueError("GPU repository requires a plain HTTPS URL")
        parsed = urlparse(url)
        if not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Invalid GPU repository URL")
        suite, components = repository["suite"], repository["components"]
        if not isinstance(suite, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", suite):
            raise ValueError("Invalid GPU repository suite")
        if not isinstance(components, list) or not all(
            isinstance(component, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", component) for component in components
        ):
            raise ValueError("Invalid GPU repository components")
        if suite.endswith("/") != (len(components) == 0):
            raise ValueError("Flat repositories require a trailing slash and no components")
        checksum = repository["keyring_sha256"]
        if not isinstance(checksum, str) or not re.fullmatch(r"[a-f0-9]{64}", checksum):
            raise ValueError("Pin the GPU repository keyring SHA-256")


def install_apt_repositories(config, payload, root=Path("/")):
    repositories = config.get("apt_repositories", [])
    validate_apt_repositories(repositories)
    for index, repository in enumerate(repositories):
        source = Path(payload) / f"inputs/gpu_apt_{index}.gpg"
        if digest(source) != repository["keyring_sha256"]:
            raise ValueError("GPU repository keyring digest mismatch")
        keyring = f"/usr/share/keyrings/cvm_gpu_{index}.gpg"
        copy_file(source, root, keyring, 0o644)
        write_file(
            root,
            f"/etc/apt/sources.list.d/cvm_gpu_{index}.sources",
            f"Types: deb\nURIs: {repository['url']}\nSuites: {repository['suite']}\n"
            f"Components: {' '.join(repository['components'])}\nArchitectures: amd64\nSigned-By: {keyring}\n",
        )


def docker_configuration(gpu):
    value = {
        "data-root": "/vault/docker/data",
        "exec-root": "/run/docker",
        "log-driver": "local",
        "storage-driver": "overlay2",
        "features": {"containerd-snapshotter": False},
        # Daemon-wide defaults; cvm_app.service repeats them on the run command.
        "no-new-privileges": True,
    }
    if gpu == "nvidia_cc":
        value["runtimes"] = {"nvidia": {"path": "nvidia-container-runtime", "runtimeArgs": []}}
    return json.dumps(value, separators=(",", ":")) + "\n"


def configure_docker_service(root):
    """Disable socket activation in the measured vendor unit before masking it.

    The stock Docker unit requires docker.socket and consumes its fd:// listener.
    Keep the vendor's other settings, but open the Unix socket directly so Docker
    can be started explicitly after bootstrap without a fourth CVM unit/drop-in.
    """
    path = target(root, "/usr/lib/systemd/system/docker.service")
    text = path.read_text()
    if " -H fd://" not in text and " -H unix:///var/run/docker.sock" not in text:
        raise ValueError("Unsupported Docker service listener configuration")
    text = text.replace(" -H fd://", " -H unix:///var/run/docker.sock")
    text = re.sub(
        r"^(Requires|Wants|After)=(.*)$",
        lambda match: match[1] + "=" + " ".join(unit for unit in match[2].split() if unit != "docker.socket"),
        text,
        flags=re.MULTILINE,
    )
    path.write_text(text)


def install_files(config, payload, root=Path("/")):
    payload = Path(payload)
    root = Path(root)
    for path in DIRECTORIES:
        target(root, path).mkdir(parents=True, exist_ok=True, mode=0o755)

    package = payload / "source/cvm"
    modules = sorted(package.rglob("*.py"))
    if not modules or not (package / "runtime/bootstrap.py").is_file():
        raise ValueError("Missing measured CVM runtime")
    for source in modules:
        relative = source.relative_to(package)
        if relative.parts[0] not in ("__init__.py", "common", "runtime"):
            raise ValueError("Unexpected package in measured guest payload")
        copy_file(source, root, "/usr/lib/cvm/cvm/" + relative.as_posix(), 0o644)
    copy_file(payload / "inputs/kbs-client", root, "/usr/lib/cvm/bin/kbs-client", 0o755)
    for name in ("kbs-ca.pem", "as-public.pem", "runtime.json"):
        copy_file(payload / "inputs" / name, root, "/etc/cvm/" + name, 0o644)
    write_file(root, "/etc/cvm_build_id", config["build_id"] + "\n")
    write_file(root, "/etc/cvm_profile_version", config["profile_version"] + "\n")

    if config["gpu"] == "nvidia_cc":
        library = target(root, "/usr/lib/x86_64-linux-gnu/libnvat.so.1")
        copy_file(payload / "inputs/libnvat.so.1", root, "/usr/lib/x86_64-linux-gnu/libnvat.so.1", 0o644)
        target(root, "/usr/lib/x86_64-linux-gnu/libnvat.so").symlink_to(library.name)

    services = payload / "source/services"
    units = sorted(services.iterdir())
    expected = {"cvm_bootstrap.service", "cvm_integrity.service", "cvm_app.service"}
    if {p.name for p in units} != expected or not all(p.is_file() for p in units):
        raise ValueError("Expected exactly the three CVM systemd units")
    for source in units:
        copy_file(source, root, "/usr/lib/systemd/system/" + source.name, 0o644)
    copy_file(payload / "inputs/nftables.conf", root, "/etc/nftables.conf", 0o644)
    configure_docker_service(root)
    mask(root, ("docker.socket",))

    write_file(root, "/etc/docker/daemon.json", docker_configuration(config["gpu"]))
    write_file(
        root, "/etc/containerd/config.toml", 'version = 2\nroot = "/vault/containerd"\nstate = "/run/containerd"\n'
    )
    for path in ("hooks/cvm_verity", "scripts/local-top/verity_root", "scripts/local-bottom/overlay_root"):
        copy_file(payload / "source/initramfs" / path, root, "/etc/initramfs-tools/" + path, 0o755)

    if config["dev_mode"]:
        write_file(root, "/etc/cvm/dev_mode", "Development only: no TEE and no KBS authorization.")
    if config.get("time_servers"):
        write_file(root, "/etc/chrony/chrony.conf", chrony_configuration(config["time_servers"]))

    module = "sev_guest" if config["platform"] == "amd_sev_snp" else "tdx_guest"
    write_file(root, "/etc/modules-load.d/cvm.conf", f"{module}\ndm_crypt\ndm_integrity\n")
    write_file(
        root,
        "/etc/fstab",
        "tmpfs /tmp tmpfs defaults,nosuid,nodev,mode=1777 0 0\n",
    )
    write_file(root, "/etc/sysctl.d/99-cvm-hardening.conf", HARDENING_SYSCTL)


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
    install_apt_repositories(config, payload)
    environment = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
    with suppress_service_starts():
        execute(["apt-get", "-o", "APT::Update::Error-Mode=any", "update"], env=environment)
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


def sudoers_files_for(user, directory=Path("/etc/sudoers.d")):
    """Return sudoers drop-ins granting rules to the build user, such as cloud-init's."""
    result = []
    directory = Path(directory)
    if not directory.is_dir():
        return result
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        for line in path.read_text(errors="replace").splitlines():
            tokens = line.split()
            if tokens and not tokens[0].startswith("#") and tokens[0] == user:
                result.append(path)
                break
    return result


def scrub_identity(root=Path("/")):
    """Remove per-build identity so guests booted from the public root do not share it."""
    root = Path(root)
    # An empty machine-id makes systemd generate a fresh one on the overlay at boot.
    write_file(root, "/etc/machine-id", "")
    dbus_id = target(root, "/var/lib/dbus/machine-id")
    if dbus_id.is_file() and not dbus_id.is_symlink():
        remove(dbus_id)
    for path in target(root, "/etc/ssh").glob("ssh_host_*"):
        remove(path)
    for path in IDENTITY_FILES:
        remove(target(root, path))
    for directory in ("/var/log/journal", "/var/log/apt", "/var/lib/apt/lists"):
        location = target(root, directory)
        if location.is_dir():
            for path in location.iterdir():
                remove(path)
    write_file(root, "/etc/hostname", "cvm\n")


def remove_build_access(config, root=Path("/")):
    """Lock, expire and disarm the construction account before deleting it."""
    user = config["build_user"]
    for path in sudoers_files_for(user, target(root, "/etc/sudoers.d")):
        remove(path)
    execute(["passwd", "-l", "root"])
    execute(["passwd", "-l", user])
    execute(["usermod", "--lock", "--expiredate", "1970-01-02", "--shell", "/usr/sbin/nologin", user])
    for path in ("/root/.ssh", f"/home/{user}/.ssh", f"/home/{user}/.bash_history"):
        remove(target(root, path))


def finalize(config, payload):
    # Each delivered guest negotiates its own time-service state. Build-time
    # NTS cookies must not hide missing key-exchange connectivity at first boot.
    execute(["systemctl", "stop", "chrony.service"])
    for path in Path("/var/lib/chrony").glob("*"):
        remove(path)
    execute(["systemctl", "disable", "ssh.service", "ssh.socket"])
    execute(["systemctl", "disable", "docker.service", "docker.socket", "containerd.service"])
    mask(Path("/"), ("ssh.service", "ssh.socket", "docker.socket", *LOGIN_UNITS, *UPDATE_UNITS, *CRASH_UNITS))
    write_file(Path("/"), "/etc/cloud/cloud-init.disabled", "")
    write_file(Path("/"), "/etc/sysctl.d/99-cvm-hardening.conf", HARDENING_SYSCTL)
    execute(["systemctl", "daemon-reload"])
    execute(["systemctl", "enable", "chrony.service", "nftables.service", "cvm_bootstrap.service"])
    remove_build_access(config)
    environment = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
    execute(["apt-get", "clean"], env=environment)
    for path in (
        "/var/lib/cloud",
        "/var/lib/docker",
        "/var/lib/containerd",
    ):
        remove(path)
    scrub_identity()
    remove(payload)
    # Best effort: the account is already locked, expired and without sudo. The
    # construction SSH session itself runs as this user, so deletion may be refused.
    subprocess.run(
        ["userdel", "--force", "--remove", config["build_user"]],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
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

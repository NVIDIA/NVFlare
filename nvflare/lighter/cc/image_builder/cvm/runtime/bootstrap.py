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

"""Measured guest bootstrap, application lifecycle and fail-closed shutdown."""

import argparse
import base64
import ipaddress
import json
import signal
import subprocess
from pathlib import Path

from ..common.contracts import HEADER_BYTES, STORAGE_PROFILE, binding
from ..common.errors import BuildError, require
from ..common.evidence import serial_frames
from ..common.firewall import firewall_rules
from ..common.io import read_json, write_json
from ..common.linux import memory_file, protect_process, run
from ..common.luks import inspect_header, scan, snapshot_header, validate_mapping
from ..common.measurements import measurements
from ..common.services import WRITABLE_APPLICATION_DIRS, validate_service
from ..common.validation import DEFAULT_CAPABILITIES, DEFAULT_PIDS_LIMIT, capabilities, validate_nfs_mount
from .attestation import authorized_key
from .audit import emit
from .gpu import readiness
from .platforms import guest_platform, local_report, verify_local_binding
from .storage import close_vault, disk_device
from .supervisor import supervise
from .systemd import notify

CONFIG = Path("/etc/cvm/runtime.json")


STATE = Path("/run/cvm")


# systemd-resolved's upstream list: the DHCP-learned servers this boot may use.
RESOLVED_UPSTREAMS = Path("/run/systemd/resolve/resolv.conf")


CLOCK_MAX_CORRECTION_SECONDS = 0.5


CLOCK_MAX_SKEW_PPM = 1000


CONTAINER_NAME = "cvm-application"


CONTAINER_STOP_SECONDS = 15


DOCKER = "/usr/bin/docker"
DOCKER_ENVIRONMENT = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/root", "LANG": "C.UTF-8"}
NFS_MOUNT = Path("/nfs_data")


MOUNT_POINTS = {"vault": "/vault", "applog": "/applog", "user-config": "/user_config", "user-data": "/user_data"}


MOUNT_OPTIONS = {
    "vault": "nosuid,nodev",
    "applog": "nosuid,nodev,noexec",
    "user-config": "ro,noload,nosuid,nodev,noexec",
    "user-data": "ro,noload,nosuid,nodev,noexec",
}


# Confinement appended to every admitted application unit. Services run as root
# unless they set User=, so restrict what that root can reach from the unit.
SERVICE_HARDENING = (
    "NoNewPrivileges=yes",
    "ProtectSystem=strict",
    "ReadWritePaths=/vault/application/runtime /vault/application/data /applog",
    "PrivateTmp=yes",
    "ProtectKernelTunables=yes",
    "ProtectKernelModules=yes",
    "ProtectKernelLogs=yes",
    "ProtectControlGroups=yes",
    "ProtectClock=yes",
    "RestrictSUIDSGID=yes",
    "LockPersonality=yes",
    "RestrictRealtime=yes",
    "CapabilityBoundingSet=~CAP_SYS_MODULE CAP_SYS_RAWIO CAP_SYS_BOOT CAP_SYS_TIME CAP_SYS_ADMIN CAP_SYS_PTRACE "
    "CAP_MAC_ADMIN CAP_MAC_OVERRIDE CAP_NET_ADMIN CAP_BPF CAP_PERFMON CAP_SYSLOG CAP_LINUX_IMMUTABLE "
    "CAP_BLOCK_SUSPEND CAP_WAKE_ALARM CAP_AUDIT_CONTROL CAP_AUDIT_READ",
)


def mount_roles(devices):
    for role, device in devices.items():
        if role == "applog":
            # This is disposable public output. Never replay a host-supplied
            # journal or trust filesystem metadata left by a previous boot.
            run(
                ["mkfs.ext4", "-q", "-F", "-O", "^has_journal", "-E", "lazy_itable_init=0,nodiscard", device],
                timeout=300,
            )
        run(["mount", "-t", "ext4", "-o", MOUNT_OPTIONS[role], device, MOUNT_POINTS[role]])


def discovered_resolvers(path=RESOLVED_UPSTREAMS):
    """Return the routable upstream DNS servers systemd-resolved learned from DHCP."""
    try:
        text = Path(path).read_text()
    except OSError:
        return []
    resolvers = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "nameserver":
            try:
                address = ipaddress.ip_address(parts[1].split("%", 1)[0])
            except ValueError:
                continue
            if not (address.is_loopback or address.is_unspecified or address.is_multicast or address.is_reserved):
                resolvers.add(str(address))
    return sorted(resolvers)


def firewall(inbound, outbound, mappings=(), inbound_sources=(), outbound_destinations=(), resolvers=()):
    rules = firewall_rules(
        inbound,
        outbound,
        mappings,
        inbound_sources=inbound_sources,
        outbound_destinations=outbound_destinations,
        resolvers=resolvers,
    )
    # Replacing our own table is atomic whether or not it already exists.
    run(["nft", "-f", "-"], input=("table inet cvm {}\ndelete table inet cvm\n" + rules).encode())


def reference():
    if Path("/etc/cvm/dev_mode").exists():
        return
    # Disk presence only skips reference logging; it never authorizes a vault.
    # Normal boot still reads a fresh report in verify_local_binding(). If udev
    # has not found the disk yet, the hardware zero-binding check below applies.
    if Path(disk_device("vault")).is_block_device():
        return
    platform = guest_platform()
    report, nonce = local_report(platform)
    # Full reports are acceptance records, not normal boot-console output.
    # Reference launches use an all-zero hardware binding and no vault disk.
    binding_bytes = report[576:624] if platform == "intel_tdx" else report[192:224]
    if any(binding_bytes):
        return
    value = {
        "platform": platform,
        "measurements": measurements(platform, report),
        "report": base64.b64encode(report).decode(),
        "nonce": base64.b64encode(nonce).decode(),
    }
    if platform == "intel_tdx":
        ccel = Path("/sys/firmware/acpi/tables/data/CCEL")
        require(ccel.is_file(), "TDX CCEL is unavailable")
        value["ccel"] = base64.b64encode(ccel.read_bytes()).decode()
    # The reference launcher's serial capture is a mode-0600 acceptance record.
    # These frames must never be forwarded to public logs or OCI artifacts.
    # CCEL can exceed journald's record limit. Use bounded serial frames with
    # compression, so console forwarding cannot silently drop the reference.

    for frame in serial_frames(value):
        print(frame, flush=True)


def time_sync(max_tries=30, *, initialize=False):
    """Require chronyd synchronization before relying on signed token time claims."""
    if Path("/etc/cvm/dev_mode").exists():
        return
    require(type(max_tries) is int and 1 <= max_tries <= 90, "Invalid clock synchronization budget")
    if initialize:
        # A sealed image has no saved frequency estimate. Align chrony's update
        # threshold with our gate, obtain an initial correction, then collect
        # extra samples. Only the final bounded skew check authorizes startup.
        run(["/usr/bin/chronyc", "maxupdateskew", str(CLOCK_MAX_SKEW_PPM)], timeout=2)
        run(
            ["/usr/bin/chronyc", "waitsync", "30", str(CLOCK_MAX_CORRECTION_SECONDS), "0", "1"],
            timeout=32,
        )
        run(["/usr/bin/chronyc", "burst", "8/16"], timeout=2)
    run(
        [
            "/usr/bin/chronyc",
            "waitsync",
            str(max_tries),
            str(CLOCK_MAX_CORRECTION_SECONDS),
            str(CLOCK_MAX_SKEW_PPM),
            "1",
        ],
        timeout=max_tries + 2,
    )


def bootstrap():
    config = read_json(CONFIG)
    STATE.mkdir(mode=0o700, exist_ok=True)
    dev = Path("/etc/cvm/dev_mode").exists()
    # Reference launches have no vault disk. Emit their frames before disk waits.
    reference()
    if not dev:
        run(["nft", "list", "table", "inet", "cvm"], timeout=5)
        # Narrow the measured bootstrap rules to the resolvers this boot learned
        # before the first KBS contact; the measured file cannot know them.
        firewall([], config["bootstrap_egress"], resolvers=discovered_resolvers())
        time_sync(max_tries=90, initialize=True)
    mount_vault(config, dev=dev)
    units = finish_bootstrap(config, dev=dev)
    supervise(config, units, STATE)


def open_vault(config, platform, device):
    """Check the attached header against local hardware, authorize, and activate the mapping."""
    header = snapshot_header(device)
    digest = binding(header)
    # No KBS contact can precede this local hardware comparison.
    verify_local_binding(platform, digest)
    with memory_file(header, sealed=True) as frozen:
        inspect_header(device, frozen)
        with authorized_key(config, digest) as key:
            run(
                [
                    "cryptsetup",
                    "open",
                    "--type",
                    "luks2",
                    "--key-file",
                    f"/proc/self/fd/{key}",
                    "--keyfile-size",
                    "64",
                    "--header",
                    f"/proc/self/fd/{frozen}",
                    device,
                    "vault",
                ],
                pass_fds=(key, frozen),
                secret=True,
            )
        actual_uuid = (
            run(
                ["cryptsetup", "luksUUID", "--header", f"/proc/self/fd/{frozen}", device],
                pass_fds=(frozen,),
                secret=True,
            )
            .decode()
            .strip()
        )
    validate_mapping("vault")
    return digest, actual_uuid


def verify_payload(config):
    """Read the authenticated payload while the independent monitor watches it."""
    if config.get("vault_prescan", True):
        scan("/dev/mapper/vault")
    require(
        run(["systemctl", "is-active", "cvm_integrity.service"]).strip() == b"active",
        "Integrity monitor stopped during scan",
    )


def check_vault_manifest(config, platform, actual_uuid):
    manifest = read_json("/vault/vault_manifest.json")
    require(
        manifest["platform"] == platform and manifest["cvm_build_id"] == config["build_id"],
        "Vault belongs to another CVM bundle",
    )
    require(
        manifest["profile_version"] == config["profile_version"]
        and manifest["storage_profile"] == STORAGE_PROFILE
        and manifest["vault_header_bytes"] == HEADER_BYTES,
        "Vault contract mismatch",
    )
    require(manifest.get("luks_uuid") == actual_uuid, "Vault UUID mismatch")
    return manifest


def mount_vault(config, dev=False):
    devices = {role: disk_device(role, wait=True) for role in ("applog", "user-config", "user-data", "vault")}
    device = devices["vault"]
    if dev:
        require(
            not Path("/dev/sev-guest").exists() and not Path("/dev/tdx_guest").exists(),
            "Dev root must not run as a TEE",
        )
        mount_roles(devices)
        manifest = read_json("/vault/vault_manifest.json")
        require(
            manifest.get("dev_mode") is True and manifest["cvm_build_id"] == config["build_id"],
            "Dev vault/bundle mismatch",
        )
        return
    protect_process()
    platform = guest_platform()
    require(platform == config["platform"], "Guest TEE does not match its measured root")
    digest, actual_uuid = open_vault(config, platform, device)
    # Type=notify, no dependency on bootstrap: wait for READY before scan.
    run(["systemctl", "start", "cvm_integrity.service"], timeout=30)
    verify_payload(config)
    mount_roles({"vault": "/dev/mapper/vault"})
    check_vault_manifest(config, platform, actual_uuid)
    mount_roles({role: device for role, device in devices.items() if role != "vault"})
    write_json(
        STATE / "binding.json",
        {
            "digest": digest.hex(),
            "luks_uuid": actual_uuid,
            "measurements": measurements(platform, local_report(platform)[0]),
        },
    )
    device = "/dev/tdx_guest" if platform == "intel_tdx" else "/dev/sev-guest"
    (STATE / "platform.env").write_text(
        f"TEE_PLATFORM={platform}\nTEE_DEVICE={device}\nTEE_DEVICE_ARGS='--device {device}'\n"
    )


def reopen():
    """Regain authorization after quarantine: same vault identity, fresh appraisal and key."""
    config = read_json(CONFIG)
    require(not Path("/etc/cvm/dev_mode").exists(), "Development images do not quarantine")
    protect_process()
    time_sync(max_tries=5)
    platform = guest_platform()
    require(platform == config["platform"], "Guest TEE does not match its measured root")
    identity = read_json(STATE / "binding.json")
    device = disk_device("vault", wait=True)
    try:
        digest, actual_uuid = open_vault(config, platform, device)
        require(
            digest.hex() == identity["digest"] and actual_uuid == identity["luks_uuid"],
            "Vault identity changed during quarantine",
        )
        verify_payload(config)
        mount_roles({"vault": "/dev/mapper/vault"})
        check_vault_manifest(config, platform, actual_uuid)
        readiness(config, True)
    except BaseException:
        # The supervisor also cleans up after killing a timed-out child, when
        # this handler cannot run. Include failures inside open_vault itself.
        close_vault()
        raise


def finish_bootstrap(config, dev=False):
    app = read_json("/vault/config/application.json")
    require(app["requires_gpu"] == (config["gpu"] == "nvidia_cc"), "Application GPU/profile mismatch")
    require(
        set(config["bootstrap_egress"]) <= set(app["allowed_out_ports"]),
        "Application firewall would disable attestation",
    )
    firewall(
        app["allowed_ports"],
        app["allowed_out_ports"],
        app["container"]["ports"],
        inbound_sources=app.get("allowed_in_cidrs", ()),
        outbound_destinations=app.get("allowed_out_cidrs", ()),
        resolvers=discovered_resolvers(),
    )
    with open("/etc/hosts", "a") as hosts:
        for hostname, address in app["hosts_entries"].items():
            hosts.write(f"\n{address} {hostname}\n")
    if dev:
        (STATE / "platform.env").write_text("TEE_PLATFORM=none\nTEE_DEVICE=\nTEE_DEVICE_ARGS=\n")
    mount_user_data()
    units = install_services()
    run(["systemctl", "daemon-reload"])
    return ["cvm_app.service", *units]


def install_services():

    destination = Path("/run/systemd/system")
    destination.mkdir(exist_ok=True)
    units = []
    source = Path("/vault/services")
    if source.exists():
        for item in sorted(source.iterdir()):
            text = item.read_text()
            executable = validate_service(item.name, text)
            require(
                executable.resolve().is_relative_to("/vault/application"), "Service executable symlink escapes payload"
            )
            require(
                not any(executable.resolve().is_relative_to(p) for p in WRITABLE_APPLICATION_DIRS),
                "Service executable cannot reside in writable application data",
            )
            deps = "cvm_bootstrap.service"
            unit = (
                text
                + "\n[Unit]\n"
                + f"After={deps} cvm_integrity.service\nRequires={deps}\nBindsTo={deps}\n"
                + "FailureAction=poweroff-force\n"
                + "[Service]\nEnvironmentFile=/run/cvm/platform.env\n"
                + "".join(line + "\n" for line in SERVICE_HARDENING)
            )
            (destination / item.name).write_text(unit)
            units.append(item.name)
    return units


def periodic():
    config = read_json(CONFIG)
    if Path("/etc/cvm/dev_mode").exists():
        return
    protect_process()
    time_sync(max_tries=5)
    digest = bytes.fromhex(read_json(STATE / "binding.json")["digest"])
    verify_local_binding(config["platform"], digest)
    # Every supervisor tick requires a fresh positive appraisal AND current key
    # authorization; a valid signature or a cached EAR is insufficient.
    with authorized_key(config, digest):
        pass

    readiness(config, True)


def docker_argv(app, *, device=None, defaults=None, environment_file=None):
    """Return the run command; container environment values use a private file descriptor."""
    cfg = app["container"]
    args = [
        DOCKER,
        "--host",
        "unix:///var/run/docker.sock",
        "run",
        "--rm",
        "--name",
        CONTAINER_NAME,
        "--network",
        "bridge",
        "--log-driver",
        "local",
        # A container escape is root in the guest and therefore vault plaintext.
        # Start from no capabilities and add back only the admitted set.
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(cfg.get("pids_limit", DEFAULT_PIDS_LIMIT)),
    ]
    for capability in capabilities(list(cfg.get("capabilities", DEFAULT_CAPABILITIES))):
        args += ["--cap-add", capability]
    if cfg.get("read_only_rootfs", True):
        args += ["--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev", "--tmpfs", "/run:rw,nosuid,nodev"]
    if cfg.get("user"):
        args += ["--user", cfg["user"]]
    for port in cfg["ports"]:
        args += ["--publish", f'{port["host"]}:{port["container"]}/tcp']
    mounts = [
        ("/vault/application", "/vault/application", True),
        ("/vault/application/runtime", "/vault/application/runtime", False),
        ("/vault/application/data", "/vault/application/data", False),
        ("/applog", "/applog", False),
        ("/user_config", "/user_config", True),
        ("/user_data", "/user_data", True),
    ]
    if cfg.get("host_bin"):
        # Opt-in only: exposes the measured root's tools to the container.
        mounts.append(("/usr/bin", "/host/bin", True))
    if app.get("nfs_mount") is not None:
        mounts.append((str(NFS_MOUNT), str(NFS_MOUNT), True))
    for source, target, ro in mounts:
        args += ["--mount", f"type=bind,source={source},target={target}" + (",readonly" if ro else "")]
    for entry in cfg["volumes"]:
        require("," not in entry["source"] + entry["target"], "Comma is not supported in mount paths")
        args += [
            "--mount",
            f'type=bind,source={entry["source"]},target={entry["target"]}'
            + (",readonly" if entry["read_only"] else ""),
        ]
    if cfg["env"]:
        require(environment_file is not None, "Container environment requires a private environment file")
        args += ["--env-file", environment_file]
    if cfg.get("tee_device"):
        require(device in ("/dev/tdx_guest", "/dev/sev-guest"), "Application requested an unavailable TEE device")
        args += ["--device", device]
    if app["requires_gpu"]:
        args += ["--gpus", "all"]
    command = cfg.get("command")
    if command == [] and "entrypoint" not in cfg:
        require(defaults is not None, "Image configuration is needed to clear its command")
        cfg = dict(cfg, entrypoint=defaults.get("Entrypoint") or [])
    if "entrypoint" in cfg:
        entry = cfg["entrypoint"]
        args += ["--entrypoint", entry[0] if entry else ""]
        # Docker CLI resets CMD when overriding ENTRYPOINT; preserve the image
        # CMD unless the application explicitly supplied a command override.
        command = entry[1:] + (command if command is not None else (defaults or {}).get("Cmd") or [])
    args += [app["image_id"]]
    if command is not None:
        args += command
    return args


def docker_environment(app):
    """Serialize container-only settings; never expose them to the privileged CLI."""
    values = app["container"]["env"]
    require(
        all(not any(c in value for c in ("\x00", "\n", "\r")) for value in values.values()),
        "Container environment values must be single-line strings",
    )
    return "".join(f"{name}={value}\n" for name, value in values.items()).encode()


def application():
    """Run the container; a requested stop exits 0, an unexpected exit keeps its status."""
    app = read_json("/vault/config/application.json")
    for name in ("runtime", "data"):
        directory = Path("/vault/application") / name
        require(directory.is_dir() and not directory.is_symlink(), "Invalid writable application directory")
    for volume in app["container"]["volumes"]:
        source = Path(volume["source"])
        require(source.resolve() == source, "Container volume source contains a symlink")
    stopping = []

    def request_stop(signum, frame):
        # systemd stops this unit with SIGTERM (KillMode=mixed). Stop the
        # container ourselves so its exit status is a requested stop, not a
        # failure that PID 1 would answer with a forced power-off.
        stopping.append(signum)
        subprocess.run(
            [
                DOCKER,
                "--host",
                "unix:///var/run/docker.sock",
                "stop",
                "--time",
                str(CONTAINER_STOP_SECONDS),
                CONTAINER_NAME,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=CONTAINER_STOP_SECONDS + 30,
            env=DOCKER_ENVIRONMENT,
        )

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    marker = Path("/vault/docker/image-loaded.json")
    expected = app["image_id"]

    def inspect():
        try:
            result = json.loads(
                run(
                    [DOCKER, "--host", "unix:///var/run/docker.sock", "image", "inspect", expected],
                    env=DOCKER_ENVIRONMENT,
                )
            )[0]
            require(result["Id"] == expected, "Loaded image ID mismatch")
            return result
        except BuildError:
            return None

    loaded = inspect()
    if not marker.is_file() or read_json(marker).get("image_id") != expected or loaded is None:
        run(
            [DOCKER, "--host", "unix:///var/run/docker.sock", "load", "--input", "/vault/docker/application.tar"],
            timeout=1800,
            env=DOCKER_ENVIRONMENT,
        )
        loaded = inspect()
        require(loaded is not None, "Archive does not contain the expected image")
        write_json(marker, {"image_id": expected})
        run(["sync", "-f", "/vault/docker"])
    if stopping:
        return 0
    config = read_json(CONFIG)
    if app["requires_gpu"]:

        readiness(config, True)
    device = {"intel_tdx": "/dev/tdx_guest", "amd_sev_snp": "/dev/sev-guest"}.get(config["platform"])
    with memory_file(docker_environment(app), sealed=True) as environment_fd:
        command = docker_argv(
            app, device=device, defaults=loaded["Config"], environment_file=f"/proc/self/fd/{environment_fd}"
        )
        process = subprocess.Popen(
            command, env=DOCKER_ENVIRONMENT, pass_fds=(environment_fd,), stdin=subprocess.DEVNULL
        )
        status = process.wait()
    if stopping:
        return 0
    return status if status >= 0 else 128 - status


def mount_user_data():
    # Only the authenticated vault can select a kernel filesystem peer. Legacy
    # clear sidecars are rejected, not silently interpreted as trusted config.
    require(
        not Path("/user_data/ext_mount.conf").exists(),
        "Move ext_mount.conf into authenticated application nfs_mount configuration",
    )
    settings = read_json("/vault/config/application.json").get("nfs_mount")
    if settings is None:
        return

    validate_nfs_mount(settings)
    # The measured root owns this path. No component or mount target comes from
    # the clear user-data disk; a hostile mnt symlink is never traversed.
    require(NFS_MOUNT.is_dir() and not NFS_MOUNT.is_symlink(), "Invalid guest-owned NFS mountpoint")
    run(
        [
            "mount",
            "-t",
            "nfs4",
            "-o",
            "ro,nosuid,nodev,noexec,sec=krb5p",
            settings["server"] + ":" + settings["export"],
            str(NFS_MOUNT),
        ],
        timeout=90,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("bootstrap", "periodic", "application", "reopen"))
    args = parser.parse_args()
    audited = args.action in ("bootstrap", "periodic", "reopen")
    try:
        status = globals()[args.action]()
    except Exception:
        if args.action == "bootstrap":
            # Ask PID 1 to enforce failure before any best-effort diagnostics.
            # It also retains the independent phase deadline if notification fails.
            try:
                notify("WATCHDOG=trigger")
            except Exception:
                pass
        if audited:

            emit("deny")
        # A traceback could include untrusted app data or token content.
        raise SystemExit("CVM " + args.action + " failed; PID 1 will power off") from None
    if args.action == "application" and status:
        raise SystemExit(status)


if __name__ == "__main__":
    main()

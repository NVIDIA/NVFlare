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

"""Measured-root bootstrap and application-neutral workload runtime."""

import argparse
import base64
import ctypes
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .attestation import authorized_key
from .common import (
    HEADER_BYTES,
    STORAGE_PROFILE,
    BuildError,
    binding,
    disk_device,
    memory_file,
    protect_process,
    read_json,
    require,
    run,
    write_json,
)
from .platforms import guest_platform, local_report, measurements, verify_local_binding
from .storage import inspect_header, scan, snapshot_header, validate_mapping

CONFIG = Path("/etc/cvm/runtime.json")
STATE = Path("/run/cvm")
CLOCK_MAX_CORRECTION_SECONDS = 0.5
CLOCK_MAX_SKEW_PPM = 1000
GPU_ATTESTATION_TIMEOUT_SECONDS = 180
MOUNT_POINTS = {"vault": "/vault", "applog": "/applog", "user-config": "/user_config", "user-data": "/user_data"}


def kernel_poweroff():
    # Python does not expose reboot(2). Call glibc's one-argument wrapper with
    # LINUX_REBOOT_CMD_POWER_OFF after systemd's forced poweroff returns.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.reboot(0x4321FEDC) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def firewall(inbound, outbound, mappings=()):
    from .config import ports

    ports(inbound)
    ports(outbound)
    # A dedicated inet table, IPv4 and IPv6. The forward chain applies the same
    # restrictions to Docker's bridge, before Docker's own permissive chains.
    rules = [
        "table inet cvm {",
        "chain input { type filter hook input priority -10; policy drop;",
        'iifname "lo" accept',
        "ct state established,related accept",
        "ip protocol icmp accept",
        "ip6 nexthdr ipv6-icmp accept",
        "udp sport 67 udp dport 68 accept",
    ]
    if inbound:
        rules.append("tcp dport { " + ",".join(map(str, inbound)) + " } accept")
    rules += [
        "}",
        "chain output { type filter hook output priority -10; policy drop;",
        'oifname "lo" accept',
        "ct state established,related accept",
        "udp dport { 53,67,123,547 } accept",
        # Ubuntu's chrony defaults use NTS key exchange before NTP traffic.
        # Keep this host time-service allowance through the application rules.
        "tcp dport { 53,4460 } accept",
        "ip protocol icmp accept",
        "ip6 nexthdr ipv6-icmp accept",
    ]
    if outbound:
        rules.append("tcp dport { " + ",".join(map(str, outbound)) + " } accept")
    rules += [
        "}",
        "chain forward { type filter hook forward priority -10; policy drop;",
        "ct state established,related accept",
        'iifname "docker0" udp dport { 53,123 } accept',
        'iifname "docker0" tcp dport 53 accept',
    ]
    if outbound:
        rules.append('iifname "docker0" tcp dport { ' + ",".join(map(str, outbound)) + " } accept")
    for mapping in mappings:
        require(mapping["host"] in inbound, "Container port is not allowed")
        ports([mapping["container"]])
        rules.append(
            f'oifname "docker0" tcp dport {mapping["container"]} ct original proto-dst {mapping["host"]} accept'
        )
    rules += ["}", "}"]
    # Replacing our own table is atomic in one nft transaction.
    existing = (
        subprocess.run(
            ["nft", "list", "table", "inet", "cvm"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
        == 0
    )
    run(["nft", "-f", "-"], input=(("delete table inet cvm\n" if existing else "") + "\n".join(rules) + "\n").encode())


def reference():
    if Path("/etc/cvm/dev_mode").exists():
        return
    platform = guest_platform()
    report, nonce = local_report(platform)
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
    # Public hardware evidence only. Never emit app configuration or secrets.
    # CCEL can exceed journald's record limit. Use bounded serial frames with
    # compression, so console forwarding cannot silently drop the reference.
    from .evidence import serial_frames

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
    devices = {role: disk_device(role, wait=True) for role in ("applog", "user-config", "user-data", "vault")}
    device = devices["vault"]
    if Path("/etc/cvm/dev_mode").exists():
        require(
            not Path("/dev/sev-guest").exists() and not Path("/dev/tdx_guest").exists(),
            "Dev root must not run as a TEE",
        )
        mounted = []
        try:
            for role in ("vault", "applog", "user-config", "user-data"):
                options = "nosuid,nodev" if role in ("vault", "applog") else "ro,noload,nosuid,nodev,noexec"
                run(["mount", "-o", options, devices[role], MOUNT_POINTS[role]])
                mounted.append(MOUNT_POINTS[role])
            manifest = read_json("/vault/vault_manifest.json")
            require(
                manifest.get("dev_mode") is True and manifest["cvm_build_id"] == config["build_id"],
                "Dev vault/bundle mismatch",
            )
            finish_bootstrap(config, dev=True)
        except Exception:
            for mount in reversed(mounted):
                subprocess.run(["umount", mount], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            raise
        return
    protect_process()
    platform = guest_platform()
    require(platform == config["platform"], "Guest TEE does not match its measured root")
    header = snapshot_header(device)
    digest = binding(header)
    # No KBS contact can precede this local hardware comparison.
    verify_local_binding(platform, digest)
    mounted = []
    try:
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
                )
            actual_uuid = (
                run(
                    ["cryptsetup", "luksUUID", "--header", f"/proc/self/fd/{frozen}", device],
                    pass_fds=(frozen,),
                )
                .decode()
                .strip()
            )
        validate_mapping("vault")
        # Type=notify, no dependency on cvm_vault: wait for READY before scan.
        run(["systemctl", "start", "cvm_integrity.service"], timeout=30)
        scan("/dev/mapper/vault")
        require(
            run(["systemctl", "is-active", "cvm_integrity.service"]).strip() == b"active",
            "Integrity monitor stopped during scan",
        )
        run(["mount", "-o", "nosuid,nodev", "/dev/mapper/vault", "/vault"])
        mounted.append("/vault")
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
        for role in ("applog", "user-config", "user-data"):
            options = "nosuid,nodev" if role == "applog" else "ro,noload,nosuid,nodev,noexec"
            run(["mount", "-o", options, devices[role], MOUNT_POINTS[role]])
            mounted.append(MOUNT_POINTS[role])
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
        finish_bootstrap(config)
    except Exception:
        subprocess.run(
            ["systemctl", "stop", "cvm_workload.target"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        for mount in reversed(mounted):
            subprocess.run(["umount", mount], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["cryptsetup", "close", "vault"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raise


def finish_bootstrap(config, dev=False):
    app = read_json("/vault/config/application.json")
    require(app["requires_gpu"] == (config["gpu"] == "nvidia_cc"), "Application GPU/profile mismatch")
    require(
        set(config["bootstrap_egress"]) <= set(app["allowed_out_ports"]),
        "Application firewall would disable attestation",
    )
    firewall(app["allowed_ports"], app["allowed_out_ports"], app["container"]["ports"])
    with open("/etc/hosts", "a") as hosts:
        for hostname, address in app["hosts_entries"].items():
            hosts.write(f"\n{address} {hostname}\n")
    if dev:
        (STATE / "platform.env").write_text("TEE_PLATFORM=none\nTEE_DEVICE=\nTEE_DEVICE_ARGS=\n")
    install_services(dev=dev)
    run(["systemctl", "daemon-reload"])
    # Workload requires this oneshot, which is still activating until we return.
    run(["systemctl", "start", "--no-block", "cvm_workload.target"])


def install_services(dev=False):
    from .services import validate_service

    destination = Path("/run/systemd/system")
    wants = destination / "cvm_workload.target.wants"
    wants.mkdir(exist_ok=True)
    source = Path("/vault/services")
    if source.exists():
        for item in sorted(source.iterdir()):
            text = item.read_text()
            validate_service(item.name, text)
            deps = "cvm_vault.service" + ("" if dev else " cvm_integrity.service")
            unit = (
                text
                + "\n[Unit]\n"
                + f"After={deps}\nRequires={deps}\nBindsTo={deps}\n"
                + "PartOf=cvm_workload.target\nOnFailure=cvm_fail.service\n"
                + "[Service]\nEnvironmentFile=/run/cvm/platform.env\n"
            )
            (destination / item.name).write_text(unit)
            (wants / item.name).symlink_to("../" + item.name)


def periodic():
    config = read_json(CONFIG)
    if Path("/etc/cvm/dev_mode").exists():
        return
    protect_process()
    time_sync(max_tries=5)
    digest = bytes.fromhex(read_json(STATE / "binding.json")["digest"])
    verify_local_binding(config["platform"], digest)
    # Every timer tick requires a fresh positive appraisal AND current key
    # authorization; a valid signature or a cached EAR is insufficient.
    with authorized_key(config, digest):
        pass
    if config["gpu"] == "nvidia_cc":
        run([sys.executable, "-m", "builder.gpu"], timeout=GPU_ATTESTATION_TIMEOUT_SECONDS)


def docker_argv(app, *, device=None, defaults=None):
    cfg = app["container"]
    args = ["docker", "run", "--rm", "--name", "cvm-application", "--network", "bridge", "--log-driver", "local"]
    for port in cfg["ports"]:
        args += ["--publish", f'{port["host"]}:{port["container"]}/tcp']
    for source, target, ro in [
        ("/vault", "/vault", False),
        ("/applog", "/applog", False),
        ("/user_config", "/user_config", True),
        ("/user_data", "/user_data", True),
        ("/usr/bin", "/host/bin", True),
    ]:
        args += ["--mount", f"type=bind,source={source},target={target}" + (",readonly" if ro else "")]
    for entry in cfg["volumes"]:
        require("," not in entry["source"] + entry["target"], "Comma is not supported in mount paths")
        args += [
            "--mount",
            f'type=bind,source={entry["source"]},target={entry["target"]}'
            + (",readonly" if entry["read_only"] else ""),
        ]
    for name, value in cfg["env"].items():
        args += ["--env", f"{name}={value}"]
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


def application():
    app = read_json("/vault/config/application.json")
    marker = Path("/vault/docker/image-loaded.json")
    expected = app["image_id"]

    def inspect():
        try:
            result = json.loads(run(["docker", "image", "inspect", expected]))[0]
            require(result["Id"] == expected, "Loaded image ID mismatch")
            return result
        except BuildError:
            return None

    loaded = inspect()
    if not marker.is_file() or read_json(marker).get("image_id") != expected or loaded is None:
        run(["docker", "load", "--input", "/vault/docker/application.tar"], timeout=1800)
        loaded = inspect()
        require(loaded is not None, "Archive does not contain the expected image")
        write_json(marker, {"image_id": expected})
        run(["sync", "-f", "/vault/docker"])
    config = read_json(CONFIG)
    if app["requires_gpu"]:
        time_sync(max_tries=5)
        run([sys.executable, "-m", "builder.gpu"], timeout=GPU_ATTESTATION_TIMEOUT_SECONDS)
    device = {"intel_tdx": "/dev/tdx_guest", "amd_sev_snp": "/dev/sev-guest"}.get(config["platform"])
    # Env values are passed through the environment, not visible in process argv.
    command = docker_argv(app, device=device, defaults=loaded["Config"])
    environment = dict(os.environ)
    for name, value in app["container"]["env"].items():
        environment[name] = value
        command[command.index(f"{name}={value}")] = name
    os.execvpe(command[0], command, environment)


def mount_user_data():
    # Optional, untrusted NFS input specification. It can select only an NFS
    # export mounted read-only at the fixed /user_data/mnt path.
    path = Path("/user_data/ext_mount.conf")
    if not path.exists():
        return
    lines = [
        line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    require(len(lines) == 1 and re.fullmatch(r"[A-Za-z0-9_.-]+:/[A-Za-z0-9_./-]+", lines[0]), "Invalid NFS input")
    require(".." not in lines[0].split(":", 1)[1].split("/"), "NFS path traversal")
    run(["mount", "-t", "nfs", "-o", "ro,nosuid,nodev,noexec,resvport", lines[0], "/user_data/mnt"], timeout=90)


def fail():
    # Stop the target synchronously, kill a stuck container process, then bypass
    # the systemd transaction queue for the final kernel poweroff operation.
    # Security failure handling must not depend on a healthy workload or manager.
    try:
        subprocess.run(
            ["systemctl", "stop", "cvm_workload.target"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        subprocess.run(
            ["systemctl", "kill", "--kill-who=all", "--signal=SIGKILL", "cvm_app.service"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        subprocess.run(
            ["sync"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        subprocess.run(
            ["systemctl", "poweroff", "--force", "--force"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    # A successful systemctl invocation normally never returns to a surviving
    # process. If it does, issue the kernel syscall directly.
    kernel_poweroff()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "bootstrap",
            "reference",
            "periodic",
            "application",
            "mount-user-data",
            "firewall",
            "time-sync",
            "fail",
        ),
    )
    args = parser.parse_args()
    audited = args.action in ("bootstrap", "periodic")
    try:
        if args.action == "firewall":
            firewall([], read_json(CONFIG)["bootstrap_egress"])
        elif args.action == "time-sync":
            time_sync(max_tries=90, initialize=True)
        else:
            globals()[args.action.replace("-", "_")]()
        if audited:
            from .audit import emit

            emit("allow")
    except Exception:
        if audited:
            from .audit import emit

            emit("deny")
        # A traceback could include untrusted app data or token content.
        raise SystemExit("CVM " + args.action + " failed; workload remains stopped") from None


if __name__ == "__main__":
    main()

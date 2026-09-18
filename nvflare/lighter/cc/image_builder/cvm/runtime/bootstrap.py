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
import json
import os
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
from ..common.validation import validate_nfs_mount
from .attestation import authorized_key
from .audit import emit
from .gpu import readiness
from .platforms import guest_platform, local_report, verify_local_binding
from .storage import disk_device
from .supervisor import supervise

CONFIG = Path("/etc/cvm/runtime.json")


STATE = Path("/run/cvm")


CLOCK_MAX_CORRECTION_SECONDS = 0.5


CLOCK_MAX_SKEW_PPM = 1000


MOUNT_POINTS = {"vault": "/vault", "applog": "/applog", "user-config": "/user_config", "user-data": "/user_data"}


MOUNT_OPTIONS = {
    "vault": "nosuid,nodev",
    "applog": "nosuid,nodev",
    "user-config": "ro,noload,nosuid,nodev,noexec",
    "user-data": "ro,noload,nosuid,nodev,noexec",
}


def mount_roles(devices):
    for role, device in devices.items():
        run(["mount", "-o", MOUNT_OPTIONS[role], device, MOUNT_POINTS[role]])


def firewall(inbound, outbound, mappings=()):
    rules = firewall_rules(inbound, outbound, mappings)
    # Replacing our own table is atomic whether or not it already exists.
    run(["nft", "-f", "-"], input=("table inet cvm {}\ndelete table inet cvm\n" + rules).encode())


def reference():
    if Path("/etc/cvm/dev_mode").exists():
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
        time_sync(max_tries=90, initialize=True)
    mount_vault(config, dev=dev)
    units = finish_bootstrap(config, dev=dev)
    supervise(config, units, STATE)


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
    # Type=notify, no dependency on bootstrap: wait for READY before scan.
    run(["systemctl", "start", "cvm_integrity.service"], timeout=30)
    scan("/dev/mapper/vault")
    require(
        run(["systemctl", "is-active", "cvm_integrity.service"]).strip() == b"active",
        "Integrity monitor stopped during scan",
    )
    mount_roles({"vault": "/dev/mapper/vault"})
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
                + f"After={deps}\nRequires={deps}\nBindsTo={deps}\n"
                + "FailureAction=poweroff-force\n"
                + "[Service]\nEnvironmentFile=/run/cvm/platform.env\n"
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


def docker_argv(app, *, device=None, defaults=None):
    cfg = app["container"]
    args = ["docker", "run", "--rm", "--name", "cvm-application", "--network", "bridge", "--log-driver", "local"]
    for port in cfg["ports"]:
        args += ["--publish", f'{port["host"]}:{port["container"]}/tcp']
    for source, target, ro in [
        ("/vault/application", "/vault/application", True),
        ("/vault/application/runtime", "/vault/application/runtime", False),
        ("/vault/application/data", "/vault/application/data", False),
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
    for name in ("runtime", "data"):
        directory = Path("/vault/application") / name
        require(directory.is_dir() and not directory.is_symlink(), "Invalid writable application directory")
    for volume in app["container"]["volumes"]:
        source = Path(volume["source"])
        require(source.resolve() == source, "Container volume source contains a symlink")
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

        readiness(config, True)
    device = {"intel_tdx": "/dev/tdx_guest", "amd_sev_snp": "/dev/sev-guest"}.get(config["platform"])
    # Env values are passed through the environment, not visible in process argv.
    command = docker_argv(app, device=device, defaults=loaded["Config"])
    environment = dict(os.environ)
    for name, value in app["container"]["env"].items():
        environment[name] = value
        command[command.index(f"{name}={value}")] = name
    os.execvpe(command[0], command, environment)


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
    run(
        [
            "mount",
            "-t",
            "nfs4",
            "-o",
            "ro,nosuid,nodev,noexec,sec=krb5p",
            settings["server"] + ":" + settings["export"],
            "/user_data/mnt",
        ],
        timeout=90,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("bootstrap", "periodic", "application"))
    args = parser.parse_args()
    audited = args.action in ("bootstrap", "periodic")
    try:
        globals()[args.action]()
        if args.action == "periodic":

            emit("allow")
    except Exception:
        if audited:

            emit("deny")
        # A traceback could include untrusted app data or token content.
        raise SystemExit("CVM " + args.action + " failed; PID 1 will power off") from None


if __name__ == "__main__":
    main()

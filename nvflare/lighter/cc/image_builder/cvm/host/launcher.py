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

"""Launch and stop CVMs, including exclusive disk and GPU ownership."""

import argparse
import contextlib
import fcntl
import hashlib
import ipaddress
import json
import os
import re
import signal
import socket
import struct
import subprocess
import time
from pathlib import Path

from ..artifacts.bundle import verify_bundle
from ..common.contracts import DISK_ROLES, qemu_binding
from ..common.errors import BuildError, require
from ..common.io import read_json, write_json
from ..common.linux import run
from ..common.references import validate_snp_policy
from .platforms import host_capabilities

PCI_ADDRESS = re.compile(r"(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")


SYSFS_PCI = Path("/sys/bus/pci/devices")


RUNTIME_ROOT = Path("/run/cvm-builder")


# Reserve enough 64-bit prefetchable MMIO space behind each root port for
# current data-center GPUs. An H800 exposes a 128 GiB BAR, so QEMU's default
# automatic bridge window is insufficient.
GPU_PREF64_RESERVE = "256G"


# Time the guest gets to power off after an ACPI power-button request before
# the launcher terminates QEMU. The guest stops the container with its normal
# grace period and syncs the vault within this window.
GRACEFUL_SHUTDOWN_SECONDS = 60


# shutdown_cvm.sh waits for the graceful window plus QEMU termination.
SHUTDOWN_WAIT_SECONDS = GRACEFUL_SHUTDOWN_SECONDS + 60


def normalize_pci(address):
    address = address.lower()
    require(PCI_ADDRESS.fullmatch(address), "Invalid GPU PCI address")
    return address if address.count(":") == 2 else "0000:" + address


def _device_value(path):
    return path.read_text().strip().lower()


def gpu_devices(sysfs=SYSFS_PCI):
    """Return NVIDIA display controllers that can be assigned through an IOMMU."""
    result = []
    for device in sorted(Path(sysfs).glob("*")):
        try:
            if (
                _device_value(device / "vendor") == "0x10de"
                and int(_device_value(device / "class"), 16) >> 16 == 0x03
                and (device / "iommu_group").exists()
            ):
                result.append(device.name.lower())
        except (OSError, ValueError):
            continue
    return result


def select_gpus(explicit=None, expected_count=None, sysfs=SYSFS_PCI):
    """Select every detected GPU, or the exact repeatable override set."""
    if explicit:
        found = [normalize_pci(address) for address in explicit]
        require(len(found) == len(set(found)), "GPU PCI addresses must be unique")
        for address in found:
            path = Path(sysfs) / address
            require(path.is_dir(), f"GPU PCI device does not exist: {address}")
            require(_device_value(path / "vendor") == "0x10de", "GPU must be an NVIDIA PCI device")
            require(int(_device_value(path / "class"), 16) >> 16 == 0x03, "PCI device is not a display controller")
            require((path / "iommu_group").exists(), "GPU is not isolated in an IOMMU group")
    else:
        found = gpu_devices(sysfs)
    require(found, "No assignable NVIDIA GPU was detected; specify --gpu PCI_ADDRESS")
    if expected_count is not None:
        require(
            type(expected_count) is int and expected_count > 0 and len(found) == expected_count,
            f"GPU profile requires exactly {expected_count} assignable NVIDIA GPU(s); detected {len(found)}",
        )
    return found


def _sysfs_write(path, value):
    with Path(path).open("w") as stream:
        stream.write(value + "\n")


@contextlib.contextmanager
def vfio_gpus(explicit=None, expected_count=None, sysfs=SYSFS_PCI):
    """Bind every function of every selected GPU slot, then restore host drivers."""
    sysfs = Path(sysfs)
    selected = select_gpus(explicit, expected_count, sysfs)
    assignments = []
    seen = set()
    for address in selected:
        slot = address.rsplit(".", 1)[0]
        functions = sorted(sysfs.glob(slot + ".*"))
        require(functions, "Selected GPU slot has no PCI functions")
        group = set()
        for function in functions:
            iommu_group = function / "iommu_group"
            require(iommu_group.exists(), "Every GPU PCI function must have an IOMMU group")
            members = sorted((iommu_group / "devices").resolve().iterdir())
            require(
                members and all(member.name.lower().rsplit(".", 1)[0] == slot for member in members),
                "GPU IOMMU group contains another PCI slot; isolate it before launch",
            )
            group.update(member.name.lower() for member in members)
        require(not seen.intersection(group), "Selected GPUs share an IOMMU group")
        seen.update(group)
        assignments.append(sorted(group))
    run(["modprobe", "vfio-pci"])
    changed = []
    try:
        for address in sorted(seen):
            device = sysfs / address
            driver_link = device / "driver"
            original = driver_link.resolve().name if driver_link.exists() else None
            if original == "vfio-pci":
                continue
            override = _device_value(device / "driver_override") if (device / "driver_override").exists() else ""
            if override == "(null)":
                override = ""
            changed.append((device, original, override))
            _sysfs_write(device / "driver_override", "vfio-pci")
            if original:
                _sysfs_write(driver_link / "unbind", device.name)
            _sysfs_write(sysfs.parent / "drivers_probe", device.name)
            require(
                driver_link.exists() and driver_link.resolve().name == "vfio-pci",
                f"Could not bind {device.name} to vfio-pci",
            )
        yield assignments
    finally:
        for device, original, override in reversed(changed):
            try:
                driver_link = device / "driver"
                if driver_link.exists():
                    _sysfs_write(driver_link / "unbind", device.name)
                _sysfs_write(device / "driver_override", override)
                if original:
                    subprocess.run(
                        ["modprobe", original], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    _sysfs_write(sysfs.parent / "drivers_probe", device.name)
            except OSError:
                # Preserve the launch result. The next launch rechecks the
                # binding, and the operator can restore a failed host driver.
                pass


def find_bundle(directory, delivery, explicit=None):
    """Use an explicit override or the delivery's embedded bundle, never a build cache."""
    bundle = Path(explicit).resolve() if explicit else directory / "cvm_bundle"
    require((bundle / "cvm_manifest.json").is_file(), "CVM bundle is missing; use --cvm-bundle")
    require(
        read_json(bundle / "cvm_manifest.json").get("build_id") == delivery["cvm_build_id"],
        "CVM bundle identity mismatch",
    )
    return bundle.resolve()


def _process_start(pid):
    data = Path(f"/proc/{pid}/stat").read_text()
    return int(data[data.rfind(")") + 2 :].split()[19])


def runtime_state_path(directory, runtime_root=None):
    root = Path(runtime_root or os.environ.get("CVM_RUNTIME_DIRECTORY", RUNTIME_ROOT))
    token = hashlib.sha256(str(Path(directory).resolve()).encode()).hexdigest()
    return root / (token + ".json")


def qmp_socket_path(directory, runtime_root=None):
    """QEMU control socket beside the root-only runtime record of this delivery."""
    return runtime_state_path(directory, runtime_root).with_suffix(".qmp")


def write_runtime_state(directory, process):
    path = runtime_state_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(not path.parent.is_symlink(), "Unsafe runtime-state directory")
    os.chmod(path.parent, 0o700)
    write_json(
        path,
        {
            "schema_version": 1,
            "vault_directory": str(Path(directory).resolve()),
            "launcher_pid": os.getpid(),
            "launcher_start": _process_start(os.getpid()),
            "qemu_pid": process.pid,
            "qemu_start": _process_start(process.pid),
        },
        mode=0o600,
    )
    return path


def require_detached(directory):
    """Detect launcher flocks and QEMU byte-range locks, including orphaned QEMU."""
    image = Path(directory) / "vault.qcow2"
    if not image.exists():
        return
    with image.open("r+b") as vault:
        try:
            fcntl.flock(vault, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.lockf(vault, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BuildError(
                "Vault is still locked by another process; inspect its QEMU owner before stopping or copying it"
            ) from None


def shutdown(directory, timeout=SHUTDOWN_WAIT_SECONDS):
    path = runtime_state_path(directory)
    require(not path.is_symlink(), "Unsafe launcher runtime state")
    if not path.is_file():
        require_detached(directory)
        raise BuildError("No CVM is running for this vault directory")
    state = read_json(path)
    pid = state.get("launcher_pid")
    start = state.get("launcher_start")
    require(type(pid) is int and type(start) is int, "Invalid launcher runtime state")
    try:
        active = _process_start(pid) == start
    except FileNotFoundError:
        active = False
    if not active:
        path.unlink(missing_ok=True)
        require_detached(directory)
        raise BuildError("No CVM is running for this vault directory") from None
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if _process_start(pid) != start:
                require_detached(directory)
                return
        except FileNotFoundError:
            require_detached(directory)
            return
        if not path.exists():
            try:
                # The record is removed just before the launcher releases its
                # vault lock. Allow that final cleanup to finish.
                require_detached(directory)
            except BuildError:
                pass
            else:
                return
        time.sleep(0.2)
    raise BuildError("Timed out waiting for the CVM to stop")


def cbit_position():
    run(["modprobe", "cpuid"])
    with open("/dev/cpu/0/cpuid", "rb", buffering=0) as stream:
        value = os.pread(stream.fileno(), 16, 0x8000001F)
    require(len(value) == 16, "Cannot read SNP CPUID")
    return struct.unpack("<IIII", value)[1] & 63


def qemu_command(
    manifest,
    bundle,
    disks,
    digest,
    *,
    gpus=None,
    host_ports=(),
    cbit=None,
    reference=False,
    qmp=None,
    bind_address="0.0.0.0",
):
    platform = manifest["platform"]
    shape = manifest["launch_shape"]
    require(platform in ("amd_sev_snp", "intel_tdx"), "Unsupported launch platform")
    require(len(disks) == (4 if reference else 5), "Invalid disk layout")
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", shape["cpu_model"]), "Invalid CPU model")
    require(
        type(shape["vcpus"]) is int
        and shape["vcpus"] > 0
        and type(shape["memory_gib"]) is int
        and shape["memory_gib"] > 0,
        "Invalid measured launch shape",
    )
    try:
        bind = ipaddress.IPv4Address(bind_address)
    except ValueError:
        raise BuildError("Forwarding bind address must be an IPv4 address") from None
    # No default devices, no VGA and no stdio monitor: the device set is exactly
    # what the manifest describes, and a terminal cannot reach the QEMU monitor.
    args = [
        "qemu-system-x86_64",
        "-enable-kvm",
        "-nodefaults",
        "-no-reboot",
        "-display",
        "none",
        "-serial",
        "stdio",
        "-monitor",
        "none",
        "-bios",
        str(Path(bundle) / "OVMF.fd"),
        "-kernel",
        str(Path(bundle) / "vmlinuz"),
        "-initrd",
        str(Path(bundle) / "initrd.img"),
        "-append",
        manifest["cmdline"],
        "-cpu",
        shape["cpu_model"],
        "-smp",
        str(shape["vcpus"]),
        "-m",
        f'{shape["memory_gib"]}G',
    ]
    if qmp is not None:
        # Control socket for an orderly ACPI power-off request from the launcher.
        args += ["-qmp", f"unix:{qmp},server=on,wait=off"]
    if shape.get("shim"):
        require(platform == "intel_tdx", "Unsupported shim boot profile")
        args += ["-shim", str(Path(bundle) / "shim.efi")]
    if manifest.get("dev_mode"):
        args += ["-machine", "q35,vmport=off"]
    elif platform == "amd_sev_snp":
        require(type(cbit) is int and 0 < cbit < 64, "SNP requires the host C-bit position")
        obj = {
            "qom-type": "sev-snp-guest",
            "id": "tee0",
            "cbitpos": cbit,
            "reduced-phys-bits": 1,
            "policy": validate_snp_policy(shape.get("snp_policy")),
            "kernel-hashes": True,
            "host-data": qemu_binding(platform, digest),
        }
        memory = {
            "qom-type": "memory-backend-memfd",
            "id": "ram0",
            "size": shape["memory_gib"] * 1024**3,
            "share": True,
            "prealloc": False,
        }
        machine = "q35,confidential-guest-support=tee0,memory-backend=ram0,vmport=off"
    else:
        obj = {"qom-type": "tdx-guest", "id": "tee0", "mrconfigid": qemu_binding(platform, digest)}
        quote = shape.get("quote_generation")
        require(
            isinstance(quote, dict) and quote.get("type") in ("vsock", "unix"),
            "TDX quote-generation socket is required",
        )
        if quote["type"] == "vsock":
            require(
                set(quote) == {"type", "cid", "port"} and type(quote["cid"]) is int and type(quote["port"]) is int,
                "Invalid QGS vsock configuration",
            )
            obj["quote-generation-socket"] = {"type": "vsock", "cid": str(quote["cid"]), "port": str(quote["port"])}
        else:
            require(set(quote) == {"type", "path"} and Path(quote["path"]).is_absolute(), "Invalid QGS Unix socket")
            obj["quote-generation-socket"] = quote
        memory = {"qom-type": "memory-backend-ram", "id": "ram0", "size": shape["memory_gib"] * 1024**3}
        machine = "q35,kernel-irqchip=split,confidential-guest-support=tee0,memory-backend=ram0,vmport=off"
    if not manifest.get("dev_mode"):
        args += ["-object", json.dumps(memory), "-object", json.dumps(obj), "-machine", machine]
    args += ["-device", "virtio-scsi-pci,id=scsi0,disable-legacy=on,iommu_platform=true,romfile="]
    for index, disk in enumerate(disks):
        readonly = index in (0, 2, 3)
        # JSON filenames cannot inject QEMU properties through commas.
        args += [
            "-blockdev",
            json.dumps(
                {
                    "driver": "file",
                    "filename": str(Path(disk).resolve()),
                    "node-name": f"file{index}",
                    "locking": "on",
                    "read-only": readonly,
                }
            ),
            "-blockdev",
            json.dumps({"driver": "qcow2", "file": f"file{index}", "node-name": f"disk{index}", "read-only": readonly}),
            "-device",
            f"scsi-hd,drive=disk{index},bus=scsi0.0,channel=0,scsi-id={index},lun=0,serial=cvm-{DISK_ROLES[index]}",
        ]
    require(all(type(port) is int and 1 <= port <= 65535 for port in host_ports), "Invalid host forwarding port")
    network = "user,id=vmnic" + "".join(f",hostfwd=tcp:{bind}:{p}-:{p}" for p in host_ports)
    args += [
        "-netdev",
        network,
        "-device",
        "virtio-net-pci,disable-legacy=on,iommu_platform=true,netdev=vmnic,romfile=",
    ]
    if manifest["contract"]["gpu"] == "nvidia_cc":
        require(
            isinstance(gpus, list)
            and len(gpus) == manifest["contract"]["gpu_count"]
            and all(isinstance(group, list) and group for group in gpus)
            and all(PCI_ADDRESS.fullmatch(address) for group in gpus for address in group),
            "GPU profile requires its configured PCI device set",
        )
        if platform == "intel_tdx":
            # Legacy VFIO tries to map all TDX private RAM and exhausts the
            # host's DMA mapping limit. IOMMUFD understands discarded/private
            # guest pages and is the supported TDX GPU assignment path.
            args += ["-object", json.dumps({"qom-type": "iommufd", "id": "iommufd0"})]
        for index, group in enumerate(gpus, 1):
            bus = f"gpu{index}"
            args += [
                "-device",
                (
                    f"pcie-root-port,id={bus},bus=pcie.0,chassis={index},slot={index},"
                    f"pref64-reserve={GPU_PREF64_RESERVE}"
                ),
            ]
            for address in group:
                vfio = f"vfio-pci,host={address},bus={bus},romfile="
                if platform == "intel_tdx":
                    vfio += ",iommufd=iommufd0"
                args += ["-device", vfio]
    else:
        require(gpus is None, "CPU-only profile cannot add a GPU")
    return args


def qmp_powerdown(path):
    """Ask QEMU for an ACPI power-button press through its control socket."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(5)
        connection.connect(str(path))
        stream = connection.makefile("rwb", buffering=0)

        def exchange(message=None):
            if message is not None:
                stream.write(json.dumps(message).encode() + b"\n")
            while True:
                line = stream.readline()
                require(line, "QMP connection closed")
                reply = json.loads(line)
                if "event" not in reply:
                    return reply

        require("QMP" in exchange(), "Unexpected QMP greeting")
        require("return" in exchange({"execute": "qmp_capabilities"}), "QMP capabilities negotiation failed")
        require("return" in exchange({"execute": "system_powerdown"}), "QMP system_powerdown rejected")


def graceful_stop(process, qmp, timeout=GRACEFUL_SHUTDOWN_SECONDS):
    """Give the guest a bounded orderly shutdown before QEMU is terminated."""
    if qmp is None or process.poll() is not None:
        return
    try:
        qmp_powerdown(qmp)
    except (OSError, ValueError, BuildError):
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


def launch(directory, bundle=None, gpu=None, bind_address="0.0.0.0"):
    directory = Path(directory).resolve()
    delivery = read_json(directory / "vault_manifest.json")
    bundle = find_bundle(directory, delivery, bundle)
    manifest = verify_bundle(bundle)
    require(
        manifest.get("dev_mode") or manifest["platform"] in host_capabilities(),
        "Host TEE does not match the sealed copy",
    )
    require(
        bool(delivery.get("dev_mode")) == bool(manifest.get("dev_mode")), "Dev and production artifacts cannot be mixed"
    )
    require(
        delivery["cvm_build_id"] == manifest["build_id"] and delivery["platform"] == manifest["platform"],
        "Wrong generic bundle for this vault",
    )
    digest = bytes(32) if manifest.get("dev_mode") else bytes.fromhex(delivery["vault_bind"])
    disks = [Path(bundle) / "verity_root.qcow2"] + [
        directory / f"{n}.qcow2" for n in ("applog", "user_config", "user_data", "vault")
    ]
    require(all(path.is_file() for path in disks), "Missing delivery disk")
    gpu_context = (
        vfio_gpus(gpu, manifest["contract"]["gpu_count"])
        if manifest["contract"]["gpu"] == "nvidia_cc"
        else contextlib.nullcontext(None)
    )
    require(manifest["contract"]["gpu"] == "nvidia_cc" or not gpu, "CPU-only profile cannot add a GPU")
    qmp = qmp_socket_path(directory)
    # Locks the actual inode, so alternate names/symlinks do not bypass ownership.
    # QEMU file-node locking remains on, including after a launcher crash.
    with gpu_context as selected_gpus, open(disks[-1], "r+b") as vault:

        try:
            fcntl.flock(vault, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BuildError("Vault is already attached; wait for the previous CVM to exit") from None
        # Only the vault owner may replace its stale control socket. A rejected
        # duplicate launch must preserve the running guest's shutdown channel.
        qmp.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        require(not qmp.parent.is_symlink(), "Unsafe runtime-state directory")
        os.chmod(qmp.parent, 0o700)
        qmp.unlink(missing_ok=True)
        command = qemu_command(
            manifest,
            bundle,
            disks,
            digest,
            gpus=selected_gpus,
            host_ports=delivery["allowed_ports"],
            cbit=cbit_position() if manifest["platform"] == "amd_sev_snp" and not manifest.get("dev_mode") else None,
            qmp=str(qmp),
            bind_address=bind_address,
        )
        try:
            return run_vm(command, directory, qmp=qmp)
        finally:
            qmp.unlink(missing_ok=True)


def run_vm(command, directory, *, qmp=None):
    """Own QEMU from spawn through exit, even when interrupted during startup."""
    process = None
    state_path = runtime_state_path(directory)
    previous = {}
    stopping = None
    cleaning = False

    def terminate(signum, frame):
        nonlocal stopping
        stopping = signum
        # Popen may still be constructing its return value. Remember the signal
        # until we own the process object; never unwind that assignment window.
        if process is not None and not cleaning:
            raise KeyboardInterrupt

    try:
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT, signal.SIGINT):
            previous[sig] = signal.signal(sig, terminate)
        if stopping is not None:
            return 128 + stopping
        # QEMU runs in its own session so a terminal Ctrl-C reaches only the
        # launcher, which then requests an orderly guest power-off. The guest
        # console is output-only; login paths on it are masked in the image.
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
        if stopping is not None:
            return 128 + stopping
        write_runtime_state(directory, process)
        result = process.wait()
        return 128 - result if result < 0 else result
    except KeyboardInterrupt:
        return 128 + (stopping or signal.SIGINT)
    finally:
        cleaning = True
        try:
            # The caller retains disk/GPU ownership until this exact child exits.
            if process is not None and process.poll() is None:
                graceful_stop(process, qmp)
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            try:
                if state_path.is_file():
                    state = read_json(state_path)
                    if state.get("launcher_pid") == os.getpid() and state.get("launcher_start") == _process_start(
                        os.getpid()
                    ):
                        state_path.unlink(missing_ok=True)
            except (OSError, ValueError, KeyError):
                pass
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description="Launch one sealed vault on its matching generic CVM")
    parser.add_argument("--cvm-bundle", help="Override automatic matching-bundle discovery")
    parser.add_argument("--vault-directory", default=".")
    parser.add_argument(
        "--gpu",
        action="append",
        help="Override automatic NVIDIA GPU detection; repeat once per required GPU PCI address",
    )
    parser.add_argument(
        "--bind-address",
        default="0.0.0.0",
        help="Host IPv4 address that forwarded application ports listen on (default: all interfaces)",
    )
    parser.add_argument("--shutdown", action="store_true", help="Stop the CVM running from this vault directory")
    args = parser.parse_args()
    try:
        if args.shutdown:
            require(args.cvm_bundle is None and args.gpu is None, "Shutdown does not accept launch overrides")
            shutdown(args.vault_directory)
            return
        raise SystemExit(launch(args.vault_directory, args.cvm_bundle, args.gpu, args.bind_address))
    except (BuildError, ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"CVM launch refused: {exc}\n")


if __name__ == "__main__":
    main()

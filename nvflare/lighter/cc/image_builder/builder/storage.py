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

"""Scoped block-device ownership and authenticated vault construction."""

import contextlib
import hashlib
import json
import os
import stat
import tempfile
import time
import uuid
from pathlib import Path

from .common import HEADER_BYTES, BuildError, canonical, digest_file, lock, require, run


def linux_root():
    require(Path("/proc/self").exists() and os.geteuid() == 0, "Disk operations require root on Linux")


@contextlib.contextmanager
def nbd(image, readonly=False):
    """Never disconnect a pre-existing NBD or touch another build's mapper."""
    linux_root()
    run(["modprobe", "nbd", "max_part=8"])
    held = contextlib.ExitStack()
    device = None
    try:
        with lock("/run/lock/cvm-builder/nbd-allocation.lock"):
            for entry in sorted(Path("/sys/block").glob("nbd*")):
                if (entry / "pid").exists():
                    continue
                candidate = "/dev/" + entry.name
                try:
                    held.enter_context(lock("/run/lock/cvm-builder/" + entry.name, blocking=False))
                except BuildError:
                    continue
                run(
                    [
                        "qemu-nbd",
                        "--format=qcow2",
                        "--connect=" + candidate,
                        *(["--read-only"] if readonly else []),
                        str(image),
                    ]
                )
                device = candidate
                break
        require(device is not None, "No free NBD device; existing connections were left intact")
        run(["udevadm", "settle"])
        for _ in range(100):
            if int(run(["blockdev", "--getsize64", device]).strip()) > 0:
                break
            time.sleep(0.05)
        else:
            raise BuildError("NBD did not become ready")
        yield device
    finally:
        if device:
            run(["qemu-nbd", "--disconnect", device])
        held.close()


@contextlib.contextmanager
def mounted(device, readonly=False):
    with tempfile.TemporaryDirectory(prefix="cvm-mount-", dir="/run") as directory:
        run(["mount", "-o", "ro,noload" if readonly else "rw", device, directory])
        try:
            yield Path(directory)
        finally:
            run(["sync", "-f", directory])
            run(["umount", directory])


def create_image(path, size_bytes):
    require(not Path(path).exists(), "Refusing to overwrite an existing image")
    run(["qemu-img", "create", "-f", "qcow2", path, str(size_bytes)])
    os.chmod(path, 0o600)


def format_vault(device, key_fd):
    run(
        [
            "cryptsetup",
            "luksFormat",
            device,
            "--batch-mode",
            "--type",
            "luks2",
            "--key-file",
            f"/proc/self/fd/{key_fd}",
            "--keyfile-size",
            "64",
            "--cipher",
            "aes-xts-random",
            "--key-size",
            "512",
            "--integrity",
            "hmac-sha256",
            "--sector-size",
            "512",
            "--offset",
            str(HEADER_BYTES // 512),
            "--luks2-metadata-size",
            "16384",
            "--luks2-keyslots-size",
            str(HEADER_BYTES - 32768),
        ],
        pass_fds=(key_fd,),
    )


def validate_luks_metadata(metadata):
    segments = metadata.get("segments", {})
    require(set(segments) == {"0"}, "Vault must contain exactly one encrypted segment")
    segment = segments["0"]
    require(segment.get("type") == "crypt" and segment.get("encryption") == "aes-xts-random", "Wrong vault cipher")
    require(
        segment.get("offset") == str(HEADER_BYTES) and segment.get("sector_size") == 512, "Wrong vault storage layout"
    )
    require(
        segment.get("integrity", {}).get("type") in ("hmac-sha256", "hmac(sha256)"),
        "Vault lacks keyed payload authentication",
    )
    require(not segment.get("flags"), "Unsupported vault segment flags")
    config = metadata.get("config", {})
    require(
        int(config.get("json_size", -1)) == 12288 and int(config.get("keyslots_size", -1)) == HEADER_BYTES - 32768,
        "Unsupported metadata/keyslot area",
    )
    require(not config.get("flags") and not config.get("requirements"), "Unsupported persistent cryptsetup options")
    slots = metadata.get("keyslots", {})
    require(set(slots) == {"0"}, "Vault requires one fixed keyslot")
    slot = slots["0"]
    require(
        slot.get("type") == "luks2" and slot.get("key_size") == 96,
        "Expected 512-bit XTS plus 256-bit HMAC key material",
    )
    area = slot.get("area", {})
    require(
        int(area.get("offset", 0)) >= 32768
        and int(area.get("offset", 0)) + int(area.get("size", HEADER_BYTES)) <= HEADER_BYTES,
        "Keyslot is outside the frozen header",
    )
    return segment


def inspect_header(device, header_fd=None):
    args = ["cryptsetup", "luksDump", "--dump-json-metadata", device]
    if header_fd is not None:
        args += ["--header", f"/proc/self/fd/{header_fd}"]
    metadata = json.loads(run(args, pass_fds=() if header_fd is None else (header_fd,)))
    validate_luks_metadata(metadata)
    return metadata


def snapshot_header(device):
    with open(device, "rb", buffering=0) as stream:
        data = stream.read(HEADER_BYTES)
    require(len(data) == HEADER_BYTES, "Truncated logical vault header")
    return data


def validate_mapping(mapper):
    """Check the activated targets, never request dmsetup --showkeys."""
    table = run(["dmsetup", "table", mapper]).decode().split()
    require(len(table) >= 9 and table[2] == "crypt", "Expected dm-crypt target")
    require(table[3] == "capi:authenc(hmac(sha256),xts(aes))-random", "Activated authenticated cipher mismatch")
    require("integrity:48:aead" in table[8:], "Missing authenticated IV/HMAC tags")
    require(not any("allow_discards" in x or "recalculate" in x for x in table), "Unsafe dm-crypt options")
    underlying = table[6]
    require(":" in underlying, "Unexpected dm-integrity device reference")
    major, minor = underlying.split(":")
    integrity = run(["dmsetup", "table", "-j", major, "-m", minor]).decode().split()
    require(len(integrity) >= 8 and integrity[2] == "integrity", "Missing dm-integrity target")
    require(integrity[5] == "48" and integrity[6] == "J", "dm-integrity must use journal mode and 48-byte tags")
    require(
        not any(x in ("recalculate", "reset_recalculate", "allow_discards") for x in integrity),
        "Unsupported dm-integrity options",
    )
    return underlying


@contextlib.contextmanager
def opened_vault(device, key_fd, *, header_fd=None, mapper=None):
    mapper = mapper or "cvm-" + uuid.uuid4().hex
    inspect_header(device, header_fd)
    args = [
        "cryptsetup",
        "open",
        "--type",
        "luks2",
        "--key-file",
        f"/proc/self/fd/{key_fd}",
        "--keyfile-size",
        "64",
        device,
        mapper,
    ]
    fds = [key_fd]
    if header_fd is not None:
        args += ["--header", f"/proc/self/fd/{header_fd}"]
        fds.append(header_fd)
    run(args, pass_fds=tuple(fds))
    try:
        validate_mapping(mapper)
        yield "/dev/mapper/" + mapper
    finally:
        run(["cryptsetup", "close", mapper])


def scan(device):
    # read() propagates EIO. A short read is normal only at the device boundary.
    size = int(run(["blockdev", "--getsize64", device]))
    total = 0
    with open(device, "rb", buffering=0) as stream:
        while total < size:
            chunk = stream.read(min(1024 * 1024, size - total))
            require(chunk, "Short authenticated scan")
            total += len(chunk)


def copy_tree(source, destination):
    """Preserve ownership, mode, xattrs and hardlinks inside authenticated storage."""
    import subprocess

    args = ["tar", "--numeric-owner", "--xattrs", "--xattrs-include=*", "--acls", "--sparse"]
    first = subprocess.Popen(
        args + ["-C", str(source), "-cf", "-", "."], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    try:
        second = subprocess.run(
            args + ["-C", str(destination), "-xf", "-"],
            stdin=first.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3600,
        )
        first.stdout.close()
        require(first.wait(timeout=30) == 0 and second.returncode == 0, "Payload copy failed")
    finally:
        if first.poll() is None:
            first.terminate()
            first.wait(timeout=30)


def content_digest(root):
    """Deterministic content/metadata digest, excluding identity and filesystem bookkeeping."""
    result = hashlib.sha256()
    links = {}
    for path in sorted(Path(root).rglob("*")):
        name = path.relative_to(root).as_posix()
        if name == "vault_manifest.json" or name == "lost+found" or name.startswith("lost+found/"):
            continue
        st = path.lstat()
        require(
            stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode),
            "Payload cannot contain device nodes, sockets or FIFOs",
        )
        attrs = {
            key: os.getxattr(path, key, follow_symlinks=False).hex()
            for key in sorted(os.listxattr(path, follow_symlinks=False))
        }
        record = {"path": name, "mode": st.st_mode, "uid": st.st_uid, "gid": st.st_gid, "xattrs": attrs}
        if path.is_symlink():
            record["target"] = os.readlink(path)
        elif path.is_file():
            record["size"] = st.st_size
            record["sha256"] = digest_file(path)
            identity = (st.st_dev, st.st_ino)
            record["hardlink"] = links.setdefault(identity, name)
        encoded = canonical(record)
        # Length framing prevents file contents or names from impersonating
        # record boundaries. Inode numbers themselves differ between copies.
        result.update(len(encoded).to_bytes(8, "big") + encoded)
    return result.hexdigest()


def sidecar(path, size_gib, source=None, public_input=False, nfs_input=False):
    """Create a clear ext4 sidecar; input sidecars are rescanned after copying."""
    create_image(path, size_gib * 1024**3)
    with nbd(path) as device:
        run(["mkfs.ext4", "-q", "-F", device])
        with mounted(device) as root:
            if source:
                copy_tree(source, root)
            if nfs_input:
                (root / "mnt").mkdir(exist_ok=True)
            if public_input:
                # Re-scan the completed image so an input changed between
                # configuration validation and copying cannot smuggle a key
                # or symbolic link onto a clear-text sidecar.
                from .config import public_sidecar

                public_sidecar(root)


def build_verity(source, output, data_gib):
    data_size = data_gib * 1024**3
    create_image(output, data_size + data_size // 16 + 128 * 1024**2)
    with nbd(output) as device:
        run(["mkfs.ext4", "-q", "-F", "-b", "4096", "-L", "cc_root", device, str(data_size // 4096)])
        with mounted(device) as root:
            copy_tree(source, root)
            require(not any((root / "vault").iterdir()), "Generic root unexpectedly contains application payload")
        result = run(
            [
                "veritysetup",
                "format",
                device,
                device,
                "--data-block-size=4096",
                "--hash-block-size=4096",
                f"--data-blocks={data_size // 4096}",
                f"--hash-offset={data_size}",
            ]
        )
        import re

        match = re.search(rb"Root hash:\s*([0-9a-f]{64})", result)
        require(match, "Missing dm-verity root hash")
        roothash = match[1].decode()
        run(["veritysetup", "verify", device, device, roothash, f"--hash-offset={data_size}"])
    return roothash, data_size

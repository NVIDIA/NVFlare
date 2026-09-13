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

"""Immutable example cache and atomic delivery to user-owned workspaces."""

import ctypes
import errno
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from nvflare.tool.examples.source import (
    MAX_FILES,
    MAX_METADATA_BYTES,
    NAME,
    PROVENANCE_FILE,
    REPOSITORY,
    ExampleError,
    GitHubSource,
    blob_sha,
    check_compatibility,
    selected_ref,
    validate_catalog,
    validate_inventory,
)

MAX_CACHE_ENTRIES = 8
LOCK_TIMEOUT = 60
_CACHE_KEY = re.compile(r"[0-9a-f]{64}")


def default_cache_dir():
    override = os.environ.get("NVFLARE_EXAMPLES_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        root = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return root / "nvflare" / "examples"


def _remove(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _publish(source, destination):
    """Rename a complete directory without replacing even an empty destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        if not hasattr(libc, "renamex_np"):
            raise OSError(errno.ENOTSUP, "The C library does not support atomic no-replace directory delivery")
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = rename(os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        args = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # AT_FDCWD, RENAME_NOREPLACE
        argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        if hasattr(libc, "renameat2"):
            rename = libc.renameat2
            rename.argtypes = argtypes
            result = rename(*args)
        else:
            # Older glibc and musl may omit the wrapper even when the kernel supports it.
            # Linux UAPI: arch/x86/entry/syscalls/syscall_64.tbl and include/uapi/asm-generic/unistd.h.
            number = {"x86_64": 316, "aarch64": 276}.get(platform.machine())
            if number is None or ctypes.sizeof(ctypes.c_void_p) != 8 or not hasattr(libc, "syscall"):
                raise OSError(errno.ENOTSUP, "Atomic no-replace directory delivery is unavailable on this Linux ABI")
            syscall = libc.syscall
            syscall.restype = ctypes.c_long
            syscall.argtypes = [ctypes.c_long, *argtypes]
            result = syscall(number, *args)
    else:
        raise OSError(errno.ENOTSUP, "Atomic example delivery is unsupported on this platform")
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


class ExampleStore:
    def __init__(self, cache_dir=None, source=None):
        self.root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self.source = source if source is not None else GitHubSource()

    @contextmanager
    def locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(str(self.root / ".lock"), timeout=LOCK_TIMEOUT):
                # Staging directories left by a killed process are never considered cache hits.
                for path in self.root.glob(".download-*"):
                    _remove(path)
                yield
        except Timeout:
            raise ExampleError(
                "EXAMPLE_CACHE_BUSY",
                "Another examples command is using the cache.",
                "Wait for that command to finish, then retry.",
            ) from None

    def clear(self):
        with self.locked():
            entries = self._entries()
            for entry in entries:
                _remove(entry)
            return {"cache_directory": str(self.root), "entries_removed": len(entries)}

    def _entries(self):
        return [path for path in self.root.iterdir() if _CACHE_KEY.fullmatch(path.name)]

    def _prune(self, keep):
        others = sorted(
            (path for path in self._entries() if path != keep), key=lambda path: path.lstat().st_mtime, reverse=True
        )
        for path in others[MAX_CACHE_ENTRIES - 1 :]:
            _remove(path)

    def _load(self, path, identity, version_info):
        try:
            manifest = path / "manifest.json"
            if path.is_symlink() or manifest.is_symlink() or manifest.stat().st_size > MAX_METADATA_BYTES:
                return None
            metadata = json.loads(manifest.read_bytes())
            if metadata["identity"] != identity:
                return None
            entries = validate_catalog({"schema_version": 1, "examples": {metadata["name"]: metadata["entry"]}})
            files = validate_inventory(metadata["files"])
            check_compatibility(entries[metadata["name"]], version_info)
            payload = path / "payload"
            if payload.is_symlink():
                return None
            actual = set()
            expected = {item["path"] for item in files}
            expected.update(
                str(parent) for item in files for parent in Path(item["path"]).parents if str(parent) != "."
            )
            for index, item in enumerate(payload.rglob("*")):
                if index >= MAX_FILES or item.relative_to(payload).as_posix() not in expected:
                    return None
                if item.is_symlink() or not (item.is_file() or item.is_dir()):
                    return None
                if item.is_file():
                    actual.add(item.relative_to(payload).as_posix())
            if actual != {item["path"] for item in files}:
                return None
            for item in files:
                target = payload / item["path"]
                # Size/mtime alone cannot detect same-size edits with restored timestamps.
                if target.stat().st_size != item["size"] or blob_sha(target.read_bytes()) != item["sha"]:
                    return None
            return metadata
        except ExampleError as error:
            if error.code == "EXAMPLE_VERSION_INCOMPATIBLE":
                raise
            return None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _download(self, cache, identity, name, version_info):
        entries = self.source.catalog(identity["commit"])
        if name not in entries:
            raise ExampleError(
                "EXAMPLE_UNKNOWN",
                f"Unknown example: {name}.",
                "Choose a supported example: " + ", ".join(sorted(entries)),
            )
        entry = entries[name]
        check_compatibility(entry, version_info)
        files = self.source.inventory(identity["commit"], entry["path"])
        metadata = {"identity": identity, "name": name, "entry": entry, "files": files}
        with tempfile.TemporaryDirectory(prefix=".download-", dir=self.root) as temporary:
            staging = Path(temporary) / "entry"
            payload = staging / "payload"
            payload.mkdir(parents=True)
            for item in files:
                data = self.source.file(identity["commit"], f"{entry['path']}/{item['path']}", item)
                target = payload / item["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(0o755 if item["mode"] == "100755" else 0o644)
            (staging / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
            _publish(staging, cache)
        return metadata

    def get(self, version_info, *, name, ref=None, destination=None, refresh=False):
        if not NAME.fullmatch(name):
            raise ExampleError("EXAMPLE_UNKNOWN", "Invalid example short name.", "Use nvflare examples get hello-pt.")
        ref = selected_ref(version_info, ref)
        # Known destinations can fail before a network request.
        if destination is not None:
            self._check_destination(Path(destination).expanduser().absolute())
        commit = self.source.resolve(ref)
        identity = {"repository": REPOSITORY, "commit": commit, "selector": name}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        with self.locked():
            cache = self.root / key
            existed = cache.exists() or cache.is_symlink()
            metadata = self._load(cache, identity, version_info) if existed and not refresh else None
            status = "cache hit" if metadata else "downloaded"
            if metadata is None:
                if existed:
                    _remove(cache)
                try:
                    metadata = self._download(cache, identity, name, version_info)
                except ExampleError as error:
                    if existed and not refresh:
                        raise ExampleError(
                            "EXAMPLE_CACHE_CORRUPT",
                            f"A corrupt cache entry was removed; repair failed: {error}",
                            error.hint + " Retry with --refresh after resolving the download problem.",
                        ) from error
                    raise
                if existed and not refresh:
                    status = "downloaded (cache repaired)"
            os.utime(cache, None)
            self._prune(cache)
            entry = metadata["entry"]
            destination = Path(destination or entry["destination"]).expanduser().absolute()
            self._check_destination(destination)
            # Do not deliver into the cache itself, where retention/clear would remove user work.
            if destination.resolve().is_relative_to(self.root.resolve()):
                raise ExampleError(
                    "INVALID_ARGS",
                    "The destination must be outside the example cache.",
                    "Choose --dest outside the cache.",
                )
            provenance = {
                "schema_version": 1,
                "repository": REPOSITORY,
                "commit": commit,
                "example": metadata["name"],
                "path": entry["path"],
                "requested_ref": ref,
                "source_url": f"https://github.com/{REPOSITORY}/tree/{commit}/{entry['path']}",
                "nvflare_version": version_info["version"],
            }
            with tempfile.TemporaryDirectory(prefix=".nvflare-example-", dir=destination.parent) as temporary:
                staged = Path(temporary) / "example"
                shutil.copytree(cache / "payload", staged)
                # Restore source modes even if a cached file's permissions were changed locally.
                for item in metadata["files"]:
                    (staged / item["path"]).chmod(0o755 if item["mode"] == "100755" else 0o644)
                (staged / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
                try:
                    _publish(staged, destination)
                except FileExistsError:
                    self._conflict(destination)
            return {
                **provenance,
                "directory": str(destination),
                "cache_status": status,
                "required_extra": entry["extra"],
                "next_command": entry["next_command"],
                "readme": str(destination / "README.md"),
            }

    @staticmethod
    def _conflict(destination):
        raise ExampleError(
            "EXAMPLE_DESTINATION_EXISTS",
            f"Destination already exists: {destination}",
            "Use --dest <new-directory>, or move the existing directory before retrying.",
        )

    @classmethod
    def _check_destination(cls, destination):
        if destination.exists() or destination.is_symlink():
            cls._conflict(destination)
        if not destination.parent.is_dir():
            raise ExampleError(
                "EXAMPLE_DESTINATION_INVALID",
                f"Destination parent does not exist: {destination.parent}",
                "Create the parent directory or choose --dest under an existing directory.",
            )

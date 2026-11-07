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

"""Keep catalog publications out of downloadable job history."""

import hmac
import io
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

from nvflare.app_common.fdn.catalog.common.payload import (
    CONTEXT_FILE,
    MANIFEST_FILE,
    PayloadError,
    job_key,
    manifest_mac,
)
from nvflare.app_common.fdn.catalog.server.fileio import write_atomically

_PRIVATE_FILES = frozenset((CONTEXT_FILE, MANIFEST_FILE, f"{CONTEXT_FILE}.tmp", f"{MANIFEST_FILE}.tmp"))
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def verify_manifest(signed: dict, key: bytes | None, job_id: str) -> dict:
    if not isinstance(signed, dict):
        raise PayloadError("manifest is not signed")
    manifest, mac = signed.get("manifest"), signed.get("mac")
    if not (key and isinstance(mac, str) and isinstance(manifest, dict)):
        raise PayloadError("manifest is not signed")
    if not hmac.compare_digest(mac, manifest_mac(job_key(key, job_id), job_id, manifest)):
        raise PayloadError("manifest signature does not match")
    return manifest


def read_archived_manifest(data: bytes, key: bytes | None, job_id: str) -> dict:
    """The filesystem store strips the run directory. Also accept older archives that retained it."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        name = next((n for n in (MANIFEST_FILE, f"{job_id}/{MANIFEST_FILE}") if n in names), None)
        if name is None:
            raise PayloadError("manifest not found")
        signed = json.loads(zf.read(name))
    return verify_manifest(signed, key, job_id)


def redact_archive(data: bytes) -> bytes | None:
    """Remove private catalog files, including legacy nested paths. None means no replacement is needed."""
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        if not any(PurePosixPath(info.filename).name in _PRIVATE_FILES for info in source.infolist()):
            return None
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as target:
            for info in source.infolist():
                if PurePosixPath(info.filename).name not in _PRIVATE_FILES:
                    target.writestr(info, source.read(info))
        return buffer.getvalue()


class CatalogArtifacts:
    """A parent-only inbox, outside local/, startup/ and every job's workspace."""

    def __init__(self, workspace_root: str):
        self._root = Path(workspace_root) / "fdn" / "catalog-results"
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _path(self, job_id: str) -> Path:
        if not isinstance(job_id, str) or not _SAFE_JOB_ID.fullmatch(job_id):
            raise ValueError("invalid job id")
        return self._root / f"{job_id}.json"

    def capture(self, job_id: str, run_dir: str, keep: bool) -> None:
        """Save the result before removing it from the run directory. A failed save leaves the archive blocked.

        Signature verification happens at ingestion. Even an invalid result must be removed before download.
        """
        path = Path(run_dir) / MANIFEST_FILE
        try:
            signed = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            pass  # an archive retry may already have captured it
        except (ValueError, UnicodeError):
            pass  # malformed JSON establishes no publication
        else:
            if keep:
                write_atomically(str(self._path(job_id)), signed)
        for name in _PRIVATE_FILES:
            (Path(run_dir) / name).unlink(missing_ok=True)

    def read(self, job_id: str, key: bytes | None) -> dict:
        return verify_manifest(json.loads(self._path(job_id).read_text(encoding="utf-8")), key, job_id)

    def capture_archive(self, job_id: str, data: bytes, key: bytes) -> None:
        manifest = read_archived_manifest(data, key, job_id)
        write_atomically(
            str(self._path(job_id)),
            {"manifest": manifest, "mac": manifest_mac(job_key(key, job_id), job_id, manifest)},
        )

    def discard(self, job_id: str) -> None:
        self._path(job_id).unlink(missing_ok=True)

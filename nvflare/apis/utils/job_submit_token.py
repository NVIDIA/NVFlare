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

import hashlib
import io
import json
import os
import posixpath
import re
import zipfile
from typing import Iterable, Optional, Tuple, Union

from nvflare.lighter.tool_consts import NVFLARE_SIG_FILE, NVFLARE_SUBMITTER_CRT_FILE

_VOLATILE_SUBMIT_ARTIFACTS = {NVFLARE_SIG_FILE, NVFLARE_SUBMITTER_CRT_FILE}
_MAX_HASH_ZIP_MEMBER_SIZE = 10 * 1024 * 1024
SUBMIT_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
SUBMIT_TOKEN_ERROR = "submit_token must be non-empty, at most 128 characters, and match ^[A-Za-z0-9._:-]{1,128}$"


def canonical_json_hash(value) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def submitter_to_dict(submitter) -> dict:
    if isinstance(submitter, dict):
        return {
            "name": submitter.get("name") or submitter.get("submitter_name") or "",
            "org": submitter.get("org") or submitter.get("submitter_org") or "",
            "role": submitter.get("role") or submitter.get("submitter_role") or "",
        }
    return {
        "name": getattr(submitter, "name", "") or getattr(submitter, "submitter_name", "") or str(submitter or ""),
        "org": getattr(submitter, "org", "") or getattr(submitter, "submitter_org", "") or "",
        "role": getattr(submitter, "role", "") or getattr(submitter, "submitter_role", "") or "",
    }


def submit_record_scope_hashes(study: str, submitter, submit_token: str) -> Tuple[str, str, str]:
    return (
        canonical_json_hash(study or ""),
        canonical_json_hash(submitter_to_dict(submitter)),
        canonical_json_hash(submit_token or ""),
    )


def validate_submit_token(submit_token: Optional[str]) -> Optional[str]:
    if submit_token is None:
        return None
    if not isinstance(submit_token, str) or not SUBMIT_TOKEN_PATTERN.fullmatch(submit_token):
        raise ValueError(SUBMIT_TOKEN_ERROR)
    return submit_token


def canonical_job_content_hash(job_content: Union[str, bytes], exclude_names: Iterable[str] = None) -> str:
    """Hash the canonical submitted job content used for submit-token retry validation.

    The caller must provide the job content root. Directory input is hashed exactly relative
    to the provided directory. Zip input may contain one wrapper directory around the job
    content; that wrapper is stripped so a normal ``zip -r job.zip job/`` archive hashes the
    same as submitting ``job/`` directly. Passing the parent of a job directory is not
    equivalent to passing the job directory itself.
    """
    exclude = set(exclude_names or _VOLATILE_SUBMIT_ARTIFACTS)
    digest = hashlib.sha256()
    for rel_path, data in _iter_canonical_job_files(job_content, exclude):
        path_bytes = rel_path.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return f"sha256:{digest.hexdigest()}"


def _iter_canonical_job_files(job_content: Union[str, bytes], exclude_names: set):
    if isinstance(job_content, bytes):
        yield from _iter_zip_bytes(job_content, exclude_names)
        return
    if not isinstance(job_content, str):
        raise TypeError(f"job_content must be bytes or str, but got {type(job_content)}")
    if os.path.isdir(job_content):
        yield from _iter_directory(job_content, exclude_names)
        return
    with open(job_content, "rb") as f:
        yield from _iter_zip_bytes(f.read(), exclude_names)


def _iter_directory(root_dir: str, exclude_names: set):
    files = []
    for root, _, names in os.walk(root_dir):
        for name in names:
            if name in exclude_names:
                continue
            full_path = os.path.join(root, name)
            if os.path.islink(full_path):
                continue
            rel_path = os.path.relpath(full_path, root_dir)
            rel_path = posixpath.join(*rel_path.split(os.sep))
            files.append((rel_path, full_path))
    for rel_path, full_path in sorted(files):
        with open(full_path, "rb") as f:
            yield rel_path, f.read()


def _iter_zip_bytes(zip_bytes: bytes, exclude_names: set):
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        files = []
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel_path = posixpath.normpath(info.filename)
            if rel_path.startswith("../") or rel_path in {".", ".."} or posixpath.isabs(rel_path):
                raise ValueError(f"zip member has unsafe path: {info.filename!r}")
            if posixpath.basename(rel_path) in exclude_names:
                continue
            # This central-directory size is attacker-controlled and only a fast-fail hint.
            # _read_zip_member_limited enforces the real bound while reading member bytes.
            if info.file_size > _MAX_HASH_ZIP_MEMBER_SIZE:
                raise ValueError(f"zip member exceeds size limit: {info.filename!r}")
            files.append((rel_path, info))
        rel_paths = [path for path, _zip_name in files]
        strip_prefix = _single_top_level_prefix(rel_paths)
        for rel_path, info in sorted(files, key=lambda item: item[0]):
            if strip_prefix:
                rel_path = rel_path[len(strip_prefix) :]
            if not rel_path:
                continue
            yield rel_path, _read_zip_member_limited(zf, info)


def _read_zip_member_limited(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    with zf.open(info, "r") as member_file:
        data = member_file.read(_MAX_HASH_ZIP_MEMBER_SIZE + 1)
    if len(data) > _MAX_HASH_ZIP_MEMBER_SIZE:
        raise ValueError(f"zip member exceeds size limit: {info.filename!r}")
    return data


def _single_top_level_prefix(rel_paths):
    first = None
    for rel_path in rel_paths:
        parts = rel_path.split("/", 1)
        if len(parts) < 2:
            return ""
        top = parts[0]
        if first is None:
            first = top
        elif first != top:
            return ""
    return f"{first}/" if first else ""

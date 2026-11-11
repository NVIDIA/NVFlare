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

"""Prepare an outputs-only job download: the declared ``job_outputs`` of the server workspace archive."""

import os
import posixpath
import shutil
import stat
import uuid
from typing import List, Optional, Tuple
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from nvflare.apis.job_def import JobMetaKey, RunStatus, job_output_matches, validate_job_outputs

_COPY_BUFFER = 1024 * 1024


def outputs_only_refusal(meta: dict) -> Tuple[Optional[str], List[str]]:
    """Why a job's download cannot be filtered (None when it can), and its declared outputs."""
    status = meta.get(JobMetaKey.STATUS.value)
    if status != RunStatus.FINISHED_COMPLETED.value:
        return f"the job did not complete successfully (status {status})", []
    declared = meta.get(JobMetaKey.JOB_OUTPUTS.value)
    if not declared:
        return f"the job declares no {JobMetaKey.JOB_OUTPUTS.value}", []
    try:
        return None, validate_job_outputs(declared)
    except (TypeError, ValueError):
        return f"the job's {JobMetaKey.JOB_OUTPUTS.value} are invalid", []


def _member_name(info: ZipInfo) -> str:
    """The member's normalized relative path, or an empty string for a member that is never extracted."""
    if info.is_dir():
        return ""
    if stat.S_ISLNK(info.external_attr >> 16):
        return ""  # a symlink entry could point outside the download
    name = info.filename.replace("\\", "/")
    if name.startswith("/") or (len(name) > 1 and name[0].isalpha() and name[1] == ":"):
        return ""
    name = posixpath.normpath(name)
    if name in (".", "..") or name.startswith("../"):
        return ""
    return name


def filter_workspace_archive(prepared: str, destination: str, outputs: List[str]) -> List[str]:
    """Write the declared entries of the ``prepared`` workspace archive to ``destination``.

    ``prepared`` is what the job store prepared for download. It may be a symlink to the stored archive, so it is
    only read and then unlinked, never written: the stored archive stays whole for later full downloads. Members keep
    their relative paths.

    Returns:
        The declared entries found in the archive, in declaration order.
    """
    found = set()
    temporary = os.path.join(os.path.dirname(destination), f".{os.path.basename(destination)}.{uuid.uuid4().hex}.tmp")
    try:
        # Members are stored, not recompressed: this runs inside the admin command, and checkpoints compress poorly.
        with ZipFile(prepared, "r") as source, ZipFile(temporary, "w", ZIP_STORED) as target:
            for info in source.infolist():
                name = _member_name(info)
                entry = job_output_matches(name, outputs) if name else ""
                if not entry:
                    continue
                member = ZipInfo(name, date_time=info.date_time)
                member.compress_type = ZIP_STORED
                member.external_attr = info.external_attr
                with source.open(info) as reader, target.open(member, "w", force_zip64=True) as writer:
                    shutil.copyfileobj(reader, writer, _COPY_BUFFER)
                found.add(entry)
        os.unlink(prepared)  # removes a symlink itself, never the archive it points to
        os.replace(temporary, destination)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    return [e for e in outputs if e in found]

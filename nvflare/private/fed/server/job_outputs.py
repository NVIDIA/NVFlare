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

"""The job's declared outputs (``job_outputs``): archived when the job ends, served by an outputs-only download.

The outputs archive is built at job end, from the files in the run folder, because that is the last moment they exist
as files: the server then stores the whole run folder as one workspace archive and deletes the folder. An
outputs-only download sends the outputs archive the way a full download sends the workspace archive, so it needs no
work before the server replies.
"""

import os
from typing import Dict, List, Optional, Tuple
from zipfile import ZipFile

from nvflare.apis.job_def import JobMetaKey, RunStatus, job_output_matches, validate_job_outputs


def declared_outputs(meta: dict) -> List[str]:
    """The job's valid declared outputs, or an empty list."""
    declared = meta.get(JobMetaKey.JOB_OUTPUTS.value)
    if not declared:
        return []
    try:
        return validate_job_outputs(declared)
    except (TypeError, ValueError):
        return []


def outputs_only_refusal(meta: dict) -> Tuple[Optional[str], List[str]]:
    """Why an outputs-only download returns the whole job (None when it need not), and the declared outputs."""
    status = meta.get(JobMetaKey.STATUS.value)
    if status != RunStatus.FINISHED_COMPLETED.value:
        return f"the job did not complete successfully (status {status})", []
    if not meta.get(JobMetaKey.JOB_OUTPUTS.value):
        return f"the job declares no {JobMetaKey.JOB_OUTPUTS.value}", []
    outputs = declared_outputs(meta)
    if not outputs:
        return f"the job's {JobMetaKey.JOB_OUTPUTS.value} are invalid", []
    return None, outputs


def archive_job_outputs(source_dirs: List[str], outputs: List[str], destination: str) -> List[str]:
    """Zip the declared outputs found in ``source_dirs`` into ``destination``, read directly from the files.

    The source folders are those archived as the job's workspace, so the paths match a full download: each folder's
    files sit at the archive root, and a later folder's file replaces an earlier one at the same path, as unzipping
    the workspace archive does. Symlinks are skipped. Nothing is written when no declared output is found.

    Returns:
        The declared entries found, in declaration order.
    """
    members: Dict[str, str] = {}  # archive path -> source file
    for source_dir in source_dirs:
        for root, _dirs, files in os.walk(source_dir):
            for name in files:
                full_path = os.path.join(root, name)
                if os.path.islink(full_path):
                    continue  # a link could point outside the run folder
                member = os.path.relpath(full_path, source_dir).replace(os.sep, "/")
                if job_output_matches(member, outputs):
                    members[member] = full_path
    if not members:
        return []
    temporary = f"{destination}.tmp"
    try:
        with ZipFile(temporary, "w") as archive:  # stored, like the workspace archive
            for member in sorted(members):
                archive.write(members[member], arcname=member)
        os.replace(temporary, destination)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    found = {job_output_matches(member, outputs) for member in members}
    return [entry for entry in outputs if entry in found]

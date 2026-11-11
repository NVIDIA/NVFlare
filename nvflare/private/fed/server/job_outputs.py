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

"""The job's declared outputs (``job_outputs``): stored apart from the rest of the workspace when the job ends.

When a job ends, the server stores its run folder as the workspace archive and deletes the folder. For a job that
declares outputs, the declared files are first moved out of the run folder into a staging folder and stored as a
separate outputs archive, so every file is stored once: the outputs in one archive, everything else in the other.
An outputs-only download sends the outputs archive the way a full download sends the workspace archive, and a full
download sends both, unzipped into the same ``workspace/`` folder.
"""

import os
import shutil
from typing import Dict, List, Optional, Tuple

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.apis.job_def import JobMetaKey, RunStatus, job_output_matches, validate_job_outputs

# Files the server's log commands read from the workspace archive. A declared one is copied into the outputs, not
# moved, so the log commands keep working. "log.json" is job_cmds.JSON_LOG_FILE_NAME.
_LOG_FILE_NAMES = {WorkspaceConstants.LOG_FILE_NAME, WorkspaceConstants.ERROR_LOG_FILE_NAME, "log.json"}


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


class StagedOutputs:
    """Declared output files staged for the outputs archive, and how to put them back."""

    def __init__(self, staging_dir: str):
        self.staging_dir = staging_dir
        self.found: List[str] = []  # the declared entries found, in declaration order
        self._staged: Dict[str, Tuple[str, bool]] = {}  # archive path -> (source file, moved)

    def restore(self):
        """Put moved files back where they came from, for when the outputs archive could not be stored."""
        for member, (source, moved) in self._staged.items():
            staged = os.path.join(self.staging_dir, member)
            if moved and os.path.exists(staged):
                os.makedirs(os.path.dirname(source), exist_ok=True)
                shutil.move(staged, source)
        self._staged.clear()
        self.found = []

    def stage(self, member: str, source: str):
        staged = os.path.join(self.staging_dir, member)
        earlier = self._staged.get(member)
        if earlier and earlier[1]:
            # a later workspace folder replaces an earlier one's file, as when the archive is unzipped: the earlier
            # file goes back to its folder, so it stays in the workspace archive
            shutil.move(staged, earlier[0])
        os.makedirs(os.path.dirname(staged), exist_ok=True)
        moved = os.path.basename(member) not in _LOG_FILE_NAMES
        if moved:
            shutil.move(source, staged)  # a rename on the same filesystem
        else:
            shutil.copyfile(source, staged)
        self._staged[member] = (source, moved)


def stage_job_outputs(source_dirs: List[str], outputs: List[str], staging_dir: str) -> StagedOutputs:
    """Move the declared outputs found in ``source_dirs`` into ``staging_dir``, at their workspace paths.

    The source folders are those archived as the job's workspace, so the paths match a full download: each folder's
    files sit at the archive root, and a later folder's file replaces an earlier one at the same path. Symlinks are
    skipped. Log files the server's log commands read are copied rather than moved. On failure, staged files are put
    back before the error is raised.
    """
    staged = StagedOutputs(staging_dir)
    try:
        for source_dir in source_dirs:
            for root, _dirs, files in os.walk(source_dir):
                if os.path.abspath(root).startswith(os.path.abspath(staging_dir)):
                    continue
                for name in files:
                    full_path = os.path.join(root, name)
                    if os.path.islink(full_path):
                        continue  # a link could point outside the run folder
                    member = os.path.relpath(full_path, source_dir).replace(os.sep, "/")
                    if job_output_matches(member, outputs):
                        staged.stage(member, full_path)
    except BaseException:
        staged.restore()
        raise
    found = {job_output_matches(member, outputs) for member in staged._staged}
    staged.found = [entry for entry in outputs if entry in found]
    return staged

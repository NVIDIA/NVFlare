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
declares outputs, the server first builds two folders of hard links next to each workspace folder: one naming the
declared output files, one naming everything else. It stores the first as the outputs archive and the second as the
workspace archive, so every file is stored once. The run folder itself is never changed until both are stored, so a
retry or a crash loses nothing: the link folders are only extra names for the same files.

An outputs-only download sends the outputs archive the way a full download sends the workspace archive, and a full
download sends both, unzipped into the same ``workspace/`` folder.
"""

import os
import shutil
from typing import List, Optional, Tuple

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.apis.job_def import JobMetaKey, RunStatus, job_output_matches, validate_job_outputs

# Files the server's log commands read from the workspace archive. A declared one is linked into both folders, so
# the log commands keep working. "log.json" is job_cmds.JSON_LOG_FILE_NAME.
_LOG_FILE_NAMES = {WorkspaceConstants.LOG_FILE_NAME, WorkspaceConstants.ERROR_LOG_FILE_NAME, "log.json"}

_OUTPUTS_LINKS = ".outputs_links"
_WORKSPACE_LINKS = ".workspace_links"


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


def _link_folder(source_dir: str, suffix: str) -> str:
    # beside the source folder, so a hard link stays on the source folder's filesystem
    source_dir = os.path.abspath(source_dir)
    return os.path.join(os.path.dirname(source_dir), f".{os.path.basename(source_dir)}{suffix}")


class SplitWorkspace:
    """Hard-link folders naming a job's declared outputs and the rest of its workspace."""

    def __init__(self, source_dirs: List[str]):
        self.output_dirs = [_link_folder(d, _OUTPUTS_LINKS) for d in source_dirs]
        self.rest_dirs = [_link_folder(d, _WORKSPACE_LINKS) for d in source_dirs]
        self.found: List[str] = []  # the declared entries found, in declaration order

    def remove(self):
        """Remove the link folders. Only links are removed; the files keep their names in the workspace folders."""
        for d in self.output_dirs + self.rest_dirs:
            shutil.rmtree(d, ignore_errors=True)


def _link(source: str, target: str):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.islink(source):
        os.symlink(os.readlink(source), target)  # kept as it is, as the workspace archive would read it
    else:
        os.link(source, target)  # another name for the same file: nothing is copied


def split_job_outputs(source_dirs: List[str], outputs: List[str]) -> Optional[SplitWorkspace]:
    """Build hard-link folders that split ``source_dirs`` into the declared outputs and the rest.

    Each source folder gets two link folders beside it, with the same relative paths, so storing the link folders
    gives the same archive paths as storing the source folders. Symlinks are not outputs: they stay with the rest.
    The log files the server's log commands read are named in both folders. The source folders are not changed.

    Returns:
        The split, or None when no declared output exists or the links cannot be made (for example across
        filesystems); the caller then stores the source folders whole, as without declared outputs.
    """
    split = SplitWorkspace(source_dirs)
    split.remove()  # left over from an interrupted attempt: links only
    found = set()
    try:
        for source_dir, output_dir, rest_dir in zip(source_dirs, split.output_dirs, split.rest_dirs):
            os.makedirs(output_dir)
            os.makedirs(rest_dir)
            for root, dirs, files in os.walk(source_dir):
                relative_root = os.path.relpath(root, source_dir)
                for name in dirs:
                    os.makedirs(os.path.join(rest_dir, relative_root, name), exist_ok=True)
                for name in files:
                    source = os.path.join(root, name)
                    relative = os.path.normpath(os.path.join(relative_root, name))
                    member = relative.replace(os.sep, "/")
                    entry = "" if os.path.islink(source) else job_output_matches(member, outputs)
                    if entry:
                        found.add(entry)
                        _link(source, os.path.join(output_dir, relative))
                    if not entry or name in _LOG_FILE_NAMES:
                        _link(source, os.path.join(rest_dir, relative))
    except OSError:
        split.remove()
        return None
    split.found = [entry for entry in outputs if entry in found]
    if not split.found:
        split.remove()
        return None
    return split

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

# Files the server's log commands read from the workspace archive, at any depth. A declared one is linked into both
# folders, so the log commands keep working; like any output, it must not change once stored.
# "log.json" is job_cmds.JSON_LOG_FILE_NAME.
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

    def __init__(self, source_dirs: List[str], outputs: List[str]):
        self.source_dirs = list(source_dirs)
        self.outputs = outputs
        self.output_dirs = [_link_folder(d, _OUTPUTS_LINKS) for d in source_dirs]
        self.rest_dirs = [_link_folder(d, _WORKSPACE_LINKS) for d in source_dirs]
        self.found: List[str] = []  # the declared entries found, in declaration order
        self._linked = [{} for _ in source_dirs]  # relative path -> identity, for each outputs folder

    def link_outputs(self):
        """Name the declared output files found in the workspace folders in the outputs folders.

        Records each linked file's identity, and the declared entries found, in declaration order. Raises OSError
        when a link cannot be made or a folder cannot be read.
        """
        found = set()
        for source_dir, output_dir, linked in zip(self.source_dirs, self.output_dirs, self._linked):
            os.makedirs(output_dir)
            for root, _dirs, files in _walk(source_dir):
                relative_root = os.path.relpath(root, source_dir)
                for name in files:
                    source = os.path.join(root, name)
                    relative = os.path.normpath(os.path.join(relative_root, name))
                    entry = _output_entry(source, relative, self.outputs)
                    if entry:
                        found.add(entry)
                        target = os.path.join(output_dir, relative)
                        _link(source, target)
                        # the file as stored: read from the link, so a replacement of the source in between is
                        # seen as a change by link_rest
                        linked[relative] = _identity(target)
        self.found = [entry for entry in self.outputs if entry in found]

    def link_rest(self):
        """Name everything but the declared outputs in the rest folders, from a fresh walk of the workspace.

        Called after the outputs are stored, so a file created or replaced meanwhile is archived as it is now. Only
        the files named in the outputs folders are left out. Raises OSError when a link cannot be made, a folder
        cannot be read, a stored output was replaced, written to or removed meanwhile, or a declared output was
        created meanwhile; the stored outputs archive would then be out of date, so the workspace is stored whole.
        It also raises when a path stored as an output in one workspace folder is in another folder's
        rest: the workspace folders share archive paths, the later folder's copy wins, and the client unzips the
        outputs archive before the workspace archive, so the split could change which copy a full download keeps.
        """
        stored = {r for linked in self._linked for r in linked if os.path.basename(r) not in _LOG_FILE_NAMES}
        for source_dir, rest_dir, linked in zip(self.source_dirs, self.rest_dirs, self._linked):
            seen = set()
            shutil.rmtree(rest_dir, ignore_errors=True)
            os.makedirs(rest_dir)
            for root, dirs, files in _walk(source_dir):
                relative_root = os.path.relpath(root, source_dir)
                for name in dirs:
                    folder = os.path.join(root, name)
                    if os.path.islink(folder):
                        _link(folder, os.path.join(rest_dir, relative_root, name))  # a linked folder stays a link
                    else:
                        os.makedirs(os.path.join(rest_dir, relative_root, name), exist_ok=True)
                for name in files:
                    source = os.path.join(root, name)
                    relative = os.path.normpath(os.path.join(relative_root, name))
                    if relative in linked:
                        seen.add(relative)
                        if _identity(source) != linked[relative]:
                            raise OSError(f"output {relative} changed after it was stored")
                        if name in _LOG_FILE_NAMES:
                            # kept in the workspace archive too, where the log commands read it
                            _link(source, os.path.join(rest_dir, relative))
                        continue  # stored in the outputs archive
                    if relative in stored:
                        raise OSError(f"{relative} is an output in one workspace folder and not in another")
                    if _output_entry(source, relative, self.outputs):
                        raise OSError(f"output {relative} was created after the outputs were stored")
                    _link(source, os.path.join(rest_dir, relative))
            if seen != set(linked):
                raise OSError("an output was removed after it was stored")

    def remove(self) -> List[str]:
        """Remove the link folders and return those that could not be removed. Only links are removed; the files
        keep their names in the workspace folders."""
        failed = []
        for d in self.output_dirs + self.rest_dirs:
            shutil.rmtree(d, ignore_errors=True)
            if os.path.lexists(d):
                failed.append(d)
        return failed


def _walk(source_dir: str):
    def _raise(error):
        raise error  # an unreadable folder: store the workspace whole rather than lose its files

    return os.walk(source_dir, onerror=_raise)


def _identity(path: str) -> Tuple[int, int, int, int]:
    # the file and its content as stored: a replacement changes the inode, an append or overwrite the size or time
    info = os.stat(path)
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _output_entry(source: str, relative: str, outputs: List[str]) -> str:
    # a symlink is never an output: it stays with the rest
    return "" if os.path.islink(source) else job_output_matches(relative.replace(os.sep, "/"), outputs)


def _link(source: str, target: str):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.islink(source):
        # point at what the original resolves to, in the untouched workspace, so it reads the same as before
        destination = os.readlink(source)
        if not os.path.isabs(destination):
            # joined, never normalized: "a/../b" through a linked folder "a" must resolve as the original does
            folder = os.path.dirname(source)
            if not os.path.isabs(folder):
                folder = os.path.join(os.getcwd(), folder)
            destination = os.path.join(folder, destination)
        os.symlink(destination, target)
    else:
        os.link(source, target)  # another name for the same file: nothing is copied


def link_job_outputs(source_dirs: List[str], outputs: List[str]) -> Optional[SplitWorkspace]:
    """Name the declared outputs found in ``source_dirs`` in hard-link folders beside them.

    Each source folder gets an outputs link folder with the same relative paths, so storing the link folders gives
    the same archive paths as storing the source folders. The source folders are not changed. Call
    ``SplitWorkspace.link_rest`` once the outputs are stored.

    Returns:
        The split, or None when no declared output exists or the links cannot be made (for example across
        filesystems); the caller then stores the source folders whole, as without declared outputs.
    """
    split = SplitWorkspace(source_dirs, outputs)
    split.remove()  # left over from an interrupted attempt: links only
    try:
        split.link_outputs()
    except OSError:
        split.remove()
        return None
    if not split.found:
        split.remove()
        return None
    return split

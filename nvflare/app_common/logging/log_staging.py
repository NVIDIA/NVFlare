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
"""Coordinate managed client-log staging with explicit job deletion."""

import errno
import os
import stat
from contextlib import contextmanager

from nvflare.app_common.logging.constants import client_log_path, client_log_staging_root


def lock_file(fd, blocking=False):
    """Take a process-held exclusive lease, released automatically on close."""
    try:
        if os.name == "nt":
            import msvcrt

            position = os.lseek(fd, 0, os.SEEK_CUR)
            try:
                # Windows locks are mandatory: lease a byte far beyond any
                # practical log size so administrators can still read the log.
                # Locking beyond EOF does not extend the file.
                os.lseek(fd, 1 << 62, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
            finally:
                os.lseek(fd, position, os.SEEK_SET)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        return True
    except OSError as ex:
        if not blocking and ex.errno in (errno.EACCES, errno.EAGAIN):
            return False
        raise


def private_directory(path):
    if os.path.islink(path):
        raise ValueError("Client log directory must not be a symlink")
    os.makedirs(path, mode=0o700, exist_ok=True)
    if os.name == "posix" and os.stat(path).st_mode & 0o022:
        raise ValueError("Client log directory must not be writable by other users")


class ClientLogStaging:
    """Leases protect open logs across receiver instances and server processes.

    One persistent mutex file coordinates stream creation/finalization with
    deletion. It is never unlinked, which would let waiters lock different
    inodes. A deletion marker lives only as long as that job's pending streams;
    it is removed with the last file.
    """

    DELETED_MARKER = ".deleted"

    def __init__(self, workspace):
        self.root = client_log_staging_root(workspace)

    @contextmanager
    def locked(self):
        private_directory(self.root)
        path = os.path.join(self.root, ".lock")
        if os.path.islink(path):
            raise ValueError("Client log lock must not be a symlink")
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0), 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("Client log lock must be a regular file")
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            lock_file(fd, blocking=True)
            yield self
        finally:
            os.close(fd)

    def _job_dir(self, job_id):
        path = client_log_path(self.root, job_id, "client", "log.txt")
        job_dir = os.path.dirname(os.path.dirname(path))
        if os.path.islink(job_dir):
            raise ValueError("Client log directory must not be a symlink")
        return job_dir

    def mark_deleted(self, job_id):
        """Record successful deletion until every leased file has drained."""
        job_dir = self._job_dir(job_id)
        if not os.path.isdir(job_dir):
            return
        path = os.path.join(job_dir, self.DELETED_MARKER)
        if os.path.islink(path):
            raise ValueError("Client log deletion marker must not be a symlink")
        fd = os.open(
            path, os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0), 0o600
        )
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("Client log deletion marker must be a regular file")
        finally:
            os.close(fd)

    def is_deleted(self, job_id):
        path = os.path.join(self._job_dir(job_id), self.DELETED_MARKER)
        if os.path.islink(path):
            raise ValueError("Client log deletion marker must not be a symlink")
        return os.path.isfile(path)

    def remove_closed_job_logs(self, job_id):
        """Remove only unleased logs. Caller must hold the staging mutex."""
        # Validate the job component and confinement before walking its tree.
        job_dir = self._job_dir(job_id)
        marker = os.path.join(job_dir, self.DELETED_MARKER)
        for directory, dirs, files in os.walk(job_dir, topdown=False, followlinks=False):
            for name in files:
                path = os.path.join(directory, name)
                if path == marker:
                    continue
                if os.path.islink(path):
                    os.unlink(path)
                    continue
                try:
                    fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
                except FileNotFoundError:
                    continue
                try:
                    if not stat.S_ISREG(os.fstat(fd).st_mode):
                        raise ValueError("Client log must be a regular file")
                    removable = lock_file(fd)
                finally:
                    os.close(fd)
                if removable:
                    os.unlink(path)
            for name in dirs:
                path = os.path.join(directory, name)
                if os.path.islink(path):
                    os.unlink(path)
            if directory == job_dir and os.listdir(job_dir) == [self.DELETED_MARKER]:
                os.unlink(marker)
            try:
                os.rmdir(directory)
            except OSError as ex:
                if ex.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    raise

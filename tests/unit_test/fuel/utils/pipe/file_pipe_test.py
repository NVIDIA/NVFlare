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

import os
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.file_pipe import FilePipe
from nvflare.fuel.utils.pipe.pipe import Message, Topic


class TestFilePipeDeferred:
    """FilePipe must not create root_path at construction time.

    Eager os.makedirs() in __init__ materialises template paths like
    {WORKSPACE}/{JOB_ID}/{SITE_NAME} as literal directories on the
    packaging machine during recipe construction (Codex review finding).
    Directory creation must be deferred to open().
    """

    def test_init_does_not_create_root_path(self, tmp_path):
        """Constructing FilePipe must not create the root_path directory."""
        root = str(tmp_path / "pipe_root")
        assert not os.path.exists(root)
        FilePipe(mode=Mode.PASSIVE, root_path=root)
        assert not os.path.exists(root), "FilePipe.__init__ must not create root_path"

    def test_init_does_not_create_template_path(self, tmp_path):
        """Template paths with curly braces must not be created as literal dirs."""
        # Simulate the recipe default path under a clean tmp_path so the test
        # is not sensitive to leftover directories from previous runs.
        template_path = str(tmp_path / "{WORKSPACE}" / "{JOB_ID}" / "{SITE_NAME}")
        FilePipe(mode=Mode.PASSIVE, root_path=template_path)
        assert not os.path.exists(template_path), "FilePipe.__init__ must not create template path as literal dir"

    def test_remove_root_false_before_open(self, tmp_path):
        """_remove_root must be False before open() is called."""
        root = str(tmp_path / "pipe_root")
        pipe = FilePipe(mode=Mode.PASSIVE, root_path=root)
        assert pipe._remove_root is False

    def test_open_creates_root_path(self, tmp_path):
        """open() must create root_path when it does not exist."""
        root = str(tmp_path / "pipe_root")
        assert not os.path.exists(root)
        pipe = FilePipe(mode=Mode.PASSIVE, root_path=root)
        pipe.open("test_pipe")
        assert os.path.exists(root)

    def test_open_sets_remove_root_when_it_created_the_dir(self, tmp_path):
        """_remove_root must be True only when open() created root_path."""
        root = str(tmp_path / "pipe_root")
        pipe = FilePipe(mode=Mode.PASSIVE, root_path=root)
        pipe.open("test_pipe")
        assert pipe._remove_root is True

    def test_open_does_not_set_remove_root_for_preexisting_dir(self, tmp_path):
        """_remove_root must remain False when root_path already existed."""
        root = str(tmp_path / "pipe_root")
        os.makedirs(root)
        pipe = FilePipe(mode=Mode.PASSIVE, root_path=root)
        pipe.open("test_pipe")
        assert pipe._remove_root is False


class TestFilePipeGetNextTOCTOU:
    """_get_next must handle TOCTOU races without raising BrokenPipeError."""

    def _make_pipe(self, tmp_path):
        root = str(tmp_path / "pipe_root")
        pipe = FilePipe(mode=Mode.PASSIVE, root_path=root)
        pipe.open("test_pipe")
        return pipe

    def test_file_disappears_during_mtime_sort_returns_none(self, tmp_path):
        """If a file disappears while os.getmtime is called during sort, _get_next returns None.

        _safe_mtime catches FileNotFoundError and returns float('inf') so the sort
        succeeds, but _read_file then also raises BrokenPipeError because the file
        is gone. _get_next must skip it and return None.
        """
        pipe = self._make_pipe(tmp_path)
        from_dir = pipe.y_path

        fake_file = os.path.join(from_dir, "fake_msg.fobs")
        open(fake_file, "w").close()

        with patch("nvflare.fuel.utils.pipe.file_pipe.os.path.getmtime", side_effect=FileNotFoundError):
            with patch.object(pipe, "_read_file", side_effect=BrokenPipeError("pipe closed")):
                result = pipe._get_next(from_dir)

        assert result is None

    def test_file_disappears_between_listdir_and_read_returns_none(self, tmp_path):
        """If a file disappears between listdir and _read_file, _get_next returns None."""
        pipe = self._make_pipe(tmp_path)
        from_dir = pipe.y_path

        fake_file = os.path.join(from_dir, "fake_msg.fobs")
        open(fake_file, "w").close()

        # Simulate the file being removed just before _read_file is called.
        original_read_file = pipe._read_file

        def disappear_then_raise(path):
            os.remove(path)
            raise BrokenPipeError("pipe closed")

        with patch.object(pipe, "_read_file", side_effect=disappear_then_raise):
            result = pipe._get_next(from_dir)

        assert result is None

    def test_all_files_race_away_returns_none(self, tmp_path):
        """When every file in the listing races away, _get_next returns None without raising."""
        pipe = self._make_pipe(tmp_path)
        from_dir = pipe.y_path

        for i in range(3):
            f = os.path.join(from_dir, f"msg_{i}.fobs")
            open(f, "w").close()

        with patch.object(pipe, "_read_file", side_effect=BrokenPipeError("pipe closed")):
            result = pipe._get_next(from_dir)

        assert result is None

    def test_get_next_does_not_suppress_broken_pipe_from_listdir(self, tmp_path):
        """_get_next must re-raise BrokenPipeError when os.listdir itself fails (real failure)."""
        pipe = self._make_pipe(tmp_path)
        from_dir = pipe.y_path

        with patch("nvflare.fuel.utils.pipe.file_pipe.os.listdir", side_effect=OSError("dir gone")):
            with pytest.raises(BrokenPipeError):
                pipe._get_next(from_dir)


class TestFilePipeSendHeartbeatTimeout:
    """FilePipe.send() must apply 600s default only for HEARTBEAT with timeout=None."""

    def _make_pipe(self, tmp_path):

        root = str(tmp_path / "pipe_root")
        pipe = FilePipe(mode=Mode.PASSIVE, root_path=root)
        pipe.open("test_pipe")
        pipe.put_f = MagicMock(return_value=True)
        return pipe

    def test_heartbeat_none_timeout_applies_600s(self, tmp_path):
        """send(HEARTBEAT, timeout=None) must pass 600.0 to put_f."""

        pipe = self._make_pipe(tmp_path)
        msg = Message.new_request(Topic.HEARTBEAT, "")
        pipe.send(msg, timeout=None)
        pipe.put_f.assert_called_once_with(msg, 600.0)

    def test_heartbeat_explicit_timeout_not_overridden(self, tmp_path):
        """send(HEARTBEAT, timeout=5.0) must not override the caller's explicit timeout."""
        pipe = self._make_pipe(tmp_path)
        msg = Message.new_request(Topic.HEARTBEAT, "")
        pipe.send(msg, timeout=5.0)
        pipe.put_f.assert_called_once_with(msg, 5.0)

    def test_heartbeat_zero_timeout_not_overridden(self, tmp_path):
        """send(HEARTBEAT, timeout=0) must not apply 600s — 0 is intentional, not unset."""
        pipe = self._make_pipe(tmp_path)
        msg = Message.new_request(Topic.HEARTBEAT, "")
        pipe.send(msg, timeout=0)
        pipe.put_f.assert_called_once_with(msg, 0)

    def test_end_none_timeout_not_changed(self, tmp_path):
        """send(END, timeout=None) must not apply 600s — only HEARTBEAT is special."""
        pipe = self._make_pipe(tmp_path)
        msg = Message.new_request(Topic.END, "")
        pipe.send(msg, timeout=None)
        pipe.put_f.assert_called_once_with(msg, None)

    def test_abort_none_timeout_not_changed(self, tmp_path):
        """send(ABORT, timeout=None) must not apply 600s."""
        pipe = self._make_pipe(tmp_path)
        msg = Message.new_request(Topic.ABORT, "")
        pipe.send(msg, timeout=None)
        pipe.put_f.assert_called_once_with(msg, None)

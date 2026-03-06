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

from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.file_pipe import FilePipe


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

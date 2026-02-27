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
import tempfile

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.app_common.streamers.file_retriever import FileRetriever


class TestFileRetrieverValidateRequest:
    """Unit tests for FileRetriever.validate_request() path traversal security."""

    @pytest.fixture
    def source_dir(self):
        """Create a temporary source directory with a known file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = os.path.join(tmpdir, "source")
            os.makedirs(source)
            allowed_file = os.path.join(source, "allowed.txt")
            with open(allowed_file, "w", encoding="utf-8") as f:
                f.write("allowed content")
            subdir = os.path.join(source, "subdir")
            os.makedirs(subdir)
            subdir_file = os.path.join(subdir, "nested.txt")
            with open(subdir_file, "w", encoding="utf-8") as f:
                f.write("nested content")
            yield source

    @pytest.fixture
    def retriever(self, source_dir):
        return FileRetriever(source_dir=source_dir, dest_dir=None)

    @pytest.fixture
    def fl_ctx(self):
        return FLContext()

    def test_valid_file_name_returns_ok(self, retriever, fl_ctx):
        """Request for a file inside source_dir must succeed."""
        request = Shareable()
        request["file_name"] = "allowed.txt"
        rc, file_path = retriever.validate_request(request, fl_ctx)
        assert rc == ReturnCode.OK
        assert file_path is not None
        assert file_path.endswith("allowed.txt")

    def test_valid_nested_path_returns_ok(self, retriever, fl_ctx):
        """Request for a file in a subdir inside source_dir must succeed."""
        request = Shareable()
        request["file_name"] = "subdir/nested.txt"
        rc, file_path = retriever.validate_request(request, fl_ctx)
        assert rc == ReturnCode.OK
        assert file_path is not None
        assert "nested.txt" in file_path

    def test_missing_file_name_returns_bad_request(self, retriever, fl_ctx):
        """Request without file_name must be rejected."""
        request = Shareable()
        rc, file_path = retriever.validate_request(request, fl_ctx)
        assert rc == ReturnCode.BAD_REQUEST_DATA
        assert file_path is None

    def test_path_traversal_parent_parent_rejected(self, retriever, fl_ctx):
        """file_name with '..' escaping source_dir must be rejected."""
        request = Shareable()
        request["file_name"] = "../../../etc/passwd"
        rc, file_path = retriever.validate_request(request, fl_ctx)
        assert rc == ReturnCode.BAD_REQUEST_DATA
        assert file_path is None

    def test_path_traversal_single_parent_rejected(self, retriever, fl_ctx):
        """file_name with single '..' escaping source_dir must be rejected."""
        request = Shareable()
        request["file_name"] = "../outside.txt"
        rc, file_path = retriever.validate_request(request, fl_ctx)
        assert rc == ReturnCode.BAD_REQUEST_DATA
        assert file_path is None

    def test_path_traversal_subdir_escape_rejected(self, retriever, fl_ctx):
        """file_name like 'subdir/../../etc/passwd' must be rejected."""
        request = Shareable()
        request["file_name"] = "subdir/../../etc/passwd"
        rc, file_path = retriever.validate_request(request, fl_ctx)
        assert rc == ReturnCode.BAD_REQUEST_DATA
        assert file_path is None

    def test_path_traversal_dot_dot_only_rejected(self, retriever, fl_ctx):
        """file_name '..' must be rejected (resolves outside source_dir)."""
        request = Shareable()
        request["file_name"] = ".."
        rc, file_path = retriever.validate_request(request, fl_ctx)
        assert rc == ReturnCode.BAD_REQUEST_DATA
        assert file_path is None

    def test_path_traversal_slash_dot_dot_rejected(self, retriever, fl_ctx):
        """file_name starting with '/..' or containing leading slash must be rejected."""
        request = Shareable()
        request["file_name"] = "/../etc/passwd"
        rc, file_path = retriever.validate_request(request, fl_ctx)
        assert rc == ReturnCode.BAD_REQUEST_DATA
        assert file_path is None

    @pytest.mark.skipif(os.name != "posix", reason="Absolute path test is Unix-specific")
    def test_absolute_path_rejected(self, retriever, fl_ctx):
        """On Unix, file_name as absolute path must be rejected (os.path.join replaces base)."""
        request = Shareable()
        request["file_name"] = "/etc/passwd"
        rc, file_path = retriever.validate_request(request, fl_ctx)
        assert rc == ReturnCode.BAD_REQUEST_DATA
        assert file_path is None

    def test_normalized_inside_source_allowed(self, retriever, fl_ctx):
        """Path that normalizes to inside source_dir (e.g. subdir/../allowed.txt) must be allowed."""
        request = Shareable()
        request["file_name"] = "subdir/../allowed.txt"
        rc, file_path = retriever.validate_request(request, fl_ctx)
        assert rc == ReturnCode.OK
        assert file_path is not None
        assert file_path.endswith("allowed.txt")

    def test_nonexistent_file_inside_source_rejected(self, retriever, fl_ctx):
        """Valid path but non-existent file must be rejected (not path traversal)."""
        request = Shareable()
        request["file_name"] = "does_not_exist.txt"
        rc, file_path = retriever.validate_request(request, fl_ctx)
        assert rc == ReturnCode.BAD_REQUEST_DATA
        assert file_path is None


class TestFileRetrieverValidateRequestRelativeSourceDir:
    """Unit tests for FileRetriever.validate_request() path traversal when source_dir is relative."""

    @pytest.fixture
    def relative_source_setup(self):
        """Create a temp dir, chdir into it, set up a relative 'source' dir with a file. Restore cwd on teardown."""
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            source_rel = "source"
            os.makedirs(source_rel, exist_ok=True)
            allowed_file = os.path.join(source_rel, "allowed.txt")
            with open(allowed_file, "w", encoding="utf-8") as f:
                f.write("allowed content")
            subdir = os.path.join(source_rel, "subdir")
            os.makedirs(subdir, exist_ok=True)
            with open(os.path.join(subdir, "nested.txt"), "w", encoding="utf-8") as f:
                f.write("nested")
            try:
                yield source_rel
            finally:
                os.chdir(old_cwd)

    @pytest.fixture
    def retriever_relative(self, relative_source_setup):
        return FileRetriever(source_dir=relative_source_setup, dest_dir=None)

    @pytest.fixture
    def fl_ctx(self):
        return FLContext()

    def test_relative_source_dir_valid_file_returns_ok(self, retriever_relative, fl_ctx):
        """With relative source_dir, request for a file inside it must succeed."""
        request = Shareable()
        request["file_name"] = "allowed.txt"
        rc, file_path = retriever_relative.validate_request(request, fl_ctx)
        assert rc == ReturnCode.OK
        assert file_path is not None
        assert file_path.endswith("allowed.txt")

    def test_relative_source_dir_valid_nested_returns_ok(self, retriever_relative, fl_ctx):
        """With relative source_dir, request for subdir file must succeed."""
        request = Shareable()
        request["file_name"] = "subdir/nested.txt"
        rc, file_path = retriever_relative.validate_request(request, fl_ctx)
        assert rc == ReturnCode.OK
        assert file_path is not None
        assert "nested.txt" in file_path

    def test_relative_source_dir_traversal_parent_rejected(self, retriever_relative, fl_ctx):
        """With relative source_dir, '..' escaping out of source must be rejected."""
        request = Shareable()
        request["file_name"] = "../outside.txt"
        rc, file_path = retriever_relative.validate_request(request, fl_ctx)
        assert rc == ReturnCode.BAD_REQUEST_DATA
        assert file_path is None

    def test_relative_source_dir_traversal_multiple_parents_rejected(self, retriever_relative, fl_ctx):
        """With relative source_dir, '../../../etc/passwd' must be rejected."""
        request = Shareable()
        request["file_name"] = "../../../etc/passwd"
        rc, file_path = retriever_relative.validate_request(request, fl_ctx)
        assert rc == ReturnCode.BAD_REQUEST_DATA
        assert file_path is None

    def test_relative_source_dir_traversal_subdir_escape_rejected(self, retriever_relative, fl_ctx):
        """With relative source_dir, 'subdir/../../outside' must be rejected."""
        request = Shareable()
        request["file_name"] = "subdir/../../outside.txt"
        rc, file_path = retriever_relative.validate_request(request, fl_ctx)
        assert rc == ReturnCode.BAD_REQUEST_DATA
        assert file_path is None

    def test_relative_source_dir_dot_dot_only_rejected(self, retriever_relative, fl_ctx):
        """With relative source_dir, file_name '..' must be rejected."""
        request = Shareable()
        request["file_name"] = ".."
        rc, file_path = retriever_relative.validate_request(request, fl_ctx)
        assert rc == ReturnCode.BAD_REQUEST_DATA
        assert file_path is None

    def test_relative_source_dir_normalized_inside_allowed(self, retriever_relative, fl_ctx):
        """With relative source_dir, path normalizing to inside source (e.g. subdir/../allowed.txt) must be allowed."""
        request = Shareable()
        request["file_name"] = "subdir/../allowed.txt"
        rc, file_path = retriever_relative.validate_request(request, fl_ctx)
        assert rc == ReturnCode.OK
        assert file_path is not None
        assert file_path.endswith("allowed.txt")


class TestFileRetrieverValidateRequestRelativeSourceDirDot:
    """Tests when source_dir is '.' (current directory)."""

    @pytest.fixture
    def dot_source_setup(self):
        """Use current directory as source; create a file in cwd and restore cwd on teardown."""
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            with open("dot_allowed.txt", "w", encoding="utf-8") as f:
                f.write("content")
            try:
                yield "."
            finally:
                os.chdir(old_cwd)

    @pytest.fixture
    def retriever_dot(self, dot_source_setup):
        return FileRetriever(source_dir=dot_source_setup, dest_dir=None)

    @pytest.fixture
    def fl_ctx(self):
        return FLContext()

    def test_dot_source_valid_file_returns_ok(self, retriever_dot, fl_ctx):
        """With source_dir '.', valid file in cwd must succeed."""
        request = Shareable()
        request["file_name"] = "dot_allowed.txt"
        rc, file_path = retriever_dot.validate_request(request, fl_ctx)
        assert rc == ReturnCode.OK
        assert file_path is not None
        assert file_path.endswith("dot_allowed.txt")

    def test_dot_source_traversal_rejected(self, retriever_dot, fl_ctx):
        """With source_dir '.', '..' escaping must be rejected."""
        request = Shareable()
        request["file_name"] = "../etc/passwd"
        rc, file_path = retriever_dot.validate_request(request, fl_ctx)
        assert rc == ReturnCode.BAD_REQUEST_DATA
        assert file_path is None

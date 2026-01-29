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
from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.streaming.download_service import DownloadStatus
from nvflare.fuel.f3.streaming.file_downloader import FileDownloadable, add_file, download_file
from nvflare.fuel.f3.streaming.obj_downloader import ObjectDownloader
from nvflare.fuel.utils.network_utils import get_open_ports


class TestFileDownloadable:
    """Test suite for FileDownloadable."""

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            content = b"Hello World! This is test content for file downloading."
            f.write(content)
            temp_path = f.name
        yield temp_path, content
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_file_downloadable_creation(self, temp_file):
        """Test creating a FileDownloadable."""
        file_path, content = temp_file
        obj = FileDownloadable(file_path)

        assert obj.name == file_path
        assert obj.size == len(content)
        assert obj.base_obj == file_path

    def test_file_downloadable_with_invalid_file(self):
        """Test that FileDownloadable raises error for non-existent file."""
        with pytest.raises(ValueError, match="does not exist or is not a valid file"):
            FileDownloadable("/nonexistent/file.txt")

    def test_file_downloadable_produce(self, temp_file):
        """Test producing chunks from FileDownloadable."""
        file_path, content = temp_file
        chunk_size = 20
        obj = FileDownloadable(file_path, chunk_size=chunk_size)

        # First produce
        rc, data, state = obj.produce({}, "receiver1")
        assert rc == "ok"
        assert data == content[:chunk_size]
        assert state["received_bytes"] == chunk_size

        # Second produce
        rc, data, state = obj.produce({"received_bytes": chunk_size}, "receiver1")
        assert rc == "ok"
        assert data == content[chunk_size : chunk_size * 2]
        assert state["received_bytes"] == chunk_size * 2

    def test_file_downloadable_downloaded_to_one_callback(self, temp_file):
        """Test that downloaded_to_one callback is called with correct parameters."""
        file_path, _ = temp_file
        callback_mock = Mock()

        obj = FileDownloadable(file_path, file_downloaded_cb=callback_mock, test_arg="test_value")

        # Call downloaded_to_one with new parameter name
        obj.downloaded_to_one(to_receiver="receiver1", status=DownloadStatus.SUCCESS)

        # Verify callback was called with correct parameters
        callback_mock.assert_called_once_with("receiver1", DownloadStatus.SUCCESS, file_path, test_arg="test_value")

    def test_file_downloadable_downloaded_to_all_callback(self, temp_file):
        """Test that downloaded_to_all callback is called."""
        file_path, _ = temp_file
        callback_mock = Mock()

        obj = FileDownloadable(file_path, file_downloaded_cb=callback_mock)

        # Call downloaded_to_all
        obj.downloaded_to_all()

        # Verify callback was called with empty strings
        callback_mock.assert_called_once_with("", "", file_path)

    def test_file_downloadable_with_custom_chunk_size(self, temp_file):
        """Test FileDownloadable with custom chunk size."""
        file_path, content = temp_file
        custom_chunk_size = 10

        obj = FileDownloadable(file_path, chunk_size=custom_chunk_size)

        rc, data, state = obj.produce({}, "receiver1")
        assert rc == "ok"
        assert len(data) == custom_chunk_size
        assert data == content[:custom_chunk_size]

    def test_file_downloadable_eof(self, temp_file):
        """Test that FileDownloadable returns EOF when complete."""
        file_path, content = temp_file
        obj = FileDownloadable(file_path)

        # Skip to end
        rc, data, state = obj.produce({"received_bytes": len(content)}, "receiver1")
        assert rc == "eof"
        assert data is None
        assert state == {}

    def test_file_downloadable_error_on_bad_state(self, temp_file):
        """Test that FileDownloadable handles bad state."""
        file_path, _ = temp_file
        obj = FileDownloadable(file_path)

        # Invalid received_bytes
        rc, data, state = obj.produce({"received_bytes": -1}, "receiver1")
        assert rc == "error"
        assert data is None
        assert state == {}

        # Non-integer received_bytes
        rc, data, state = obj.produce({"received_bytes": "invalid"}, "receiver1")
        assert rc == "error"
        assert data is None
        assert state == {}


class TestFileDownloaderIntegration:
    """Integration tests for file downloading."""

    @pytest.fixture
    def port(self):
        return get_open_ports(1)[0]

    @pytest.fixture
    def cell(self, port, request):
        """Create a unique cell for each test."""
        test_name = request.node.name
        cell_name = f"test_cell_{test_name}_{port}"
        listening_url = f"tcp://localhost:{port}"
        cell = CoreCell(cell_name, listening_url, secure=False, credentials={})
        cell.start()
        yield cell
        cell.stop()

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            content = b"Integration test file content with multiple chunks of data."
            f.write(content)
            temp_path = f.name
        yield temp_path, content
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_add_file_to_downloader(self, cell, temp_file):
        """Test adding a file to ObjectDownloader."""
        file_path, _ = temp_file
        callback_mock = Mock()

        downloader = ObjectDownloader(cell=cell, timeout=10.0, num_receivers=2)

        ref_id = add_file(
            downloader=downloader,
            file_name=file_path,
            chunk_size=20,
            file_downloaded_cb=callback_mock,
            test_arg="test_value",
        )

        assert ref_id is not None
        assert ref_id.startswith("R")

        # Cleanup
        downloader.delete_transaction()

    def test_add_file_with_custom_ref_id(self, cell, temp_file):
        """Test adding a file with custom ref_id."""
        file_path, _ = temp_file

        downloader = ObjectDownloader(cell=cell, timeout=10.0, num_receivers=2)

        custom_ref_id = "CUSTOM_FILE_REF"
        ref_id = add_file(downloader=downloader, file_name=file_path, ref_id=custom_ref_id)

        assert ref_id == custom_ref_id

        # Cleanup
        downloader.delete_transaction()

    def test_file_callback_called_on_download_completion(self, cell, temp_file):
        """Test that file_downloaded_cb is called when download completes."""
        file_path, _ = temp_file
        callback_mock = Mock()

        downloader = ObjectDownloader(cell=cell, timeout=10.0, num_receivers=1)

        ref_id = add_file(downloader=downloader, file_name=file_path, chunk_size=20, file_downloaded_cb=callback_mock)

        # Simulate download by producing all chunks
        from nvflare.fuel.f3.streaming.download_service import DownloadService

        ref = DownloadService._ref_table.get(ref_id)
        assert ref is not None

        # Simulate download completion - this triggers downloaded_to_one
        ref.obj_downloaded(to_receiver="receiver1", status=DownloadStatus.SUCCESS)

        # Verify callback was called twice (once for downloaded_to_one, once for downloaded_to_all)
        assert callback_mock.call_count == 2

        # Check the first call (downloaded_to_one with real receiver)
        all_calls = callback_mock.call_args_list
        first_call_args = all_calls[0][0]
        assert first_call_args[0] == "receiver1"  # to_receiver parameter
        assert first_call_args[1] == DownloadStatus.SUCCESS
        assert first_call_args[2] == file_path

        # Check the second call (downloaded_to_all with empty strings)
        second_call_args = all_calls[1][0]
        assert second_call_args[0] == ""  # empty to_receiver
        assert second_call_args[1] == ""  # empty status
        assert second_call_args[2] == file_path

        # Cleanup
        downloader.delete_transaction()

    def test_file_callback_signature_compatibility(self, cell, temp_file):
        """Test that callback signature matches documentation."""
        file_path, _ = temp_file

        # Define callback matching documented signature
        def file_downloaded_callback(to_receiver: str, status: str, file_name: str, **kwargs):
            assert isinstance(to_receiver, str)
            assert isinstance(status, str)
            assert isinstance(file_name, str)
            assert file_name == file_path

        downloader = ObjectDownloader(cell=cell, timeout=10.0, num_receivers=1)

        ref_id = add_file(downloader=downloader, file_name=file_path, file_downloaded_cb=file_downloaded_callback)

        from nvflare.fuel.f3.streaming.download_service import DownloadService

        ref = DownloadService._ref_table.get(ref_id)

        # This should not raise any errors
        ref.obj_downloaded(to_receiver="receiver1", status=DownloadStatus.SUCCESS)

        # Cleanup
        downloader.delete_transaction()

    def test_download_file_to_temp_location(self, temp_file):
        """Test download_file function creates file in temp location."""
        # Note: This test doesn't actually perform network download
        # It just tests the function signature and basic setup
        file_path, _ = temp_file

        # We can't easily test the full download without setting up a full cell network
        # But we can verify the function exists and has correct signature
        assert callable(download_file)

        # Verify it accepts the expected parameters
        import inspect

        sig = inspect.signature(download_file)
        params = list(sig.parameters.keys())
        assert "from_fqcn" in params
        assert "ref_id" in params
        assert "per_request_timeout" in params
        assert "cell" in params
        assert "location" in params

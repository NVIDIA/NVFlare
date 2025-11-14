# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from nvflare.app_opt.kaggle.download_data import download


class TestDownloadData:
    """Test cases for the download function"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_dataset_name = "test/dataset"

    def teardown_method(self):
        """Clean up after each test method"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch("nvflare.app_opt.kaggle.download_data.kagglehub")
    def test_download_new_directory(self, mock_kagglehub):
        """Test downloading to a new directory (first time)"""
        # Setup mock
        mock_download_path = Path(self.temp_dir) / "mock_download"
        mock_download_path.mkdir()
        (mock_download_path / "test_file.csv").touch()
        mock_kagglehub.dataset_download.return_value = str(mock_download_path)

        output_path = Path(self.temp_dir) / "output"

        # Execute
        download(self.test_dataset_name, str(output_path))

        # Verify
        assert output_path.exists()
        assert output_path.is_dir()
        assert (output_path / ".kaggle_download_marker").exists()
        assert (output_path / "test_file.csv").exists()
        mock_kagglehub.dataset_download.assert_called_once_with(self.test_dataset_name)

    @patch("nvflare.app_opt.kaggle.download_data.kagglehub")
    def test_download_existing_directory_without_overwrite(self, mock_kagglehub):
        """Test downloading when directory exists without overwrite flag (should fail)"""
        # Setup
        output_path = Path(self.temp_dir) / "output"
        output_path.mkdir()
        (output_path / ".kaggle_download_marker").touch()

        mock_download_path = Path(self.temp_dir) / "mock_download"
        mock_download_path.mkdir()
        mock_kagglehub.dataset_download.return_value = str(mock_download_path)

        # Execute and verify
        with pytest.raises(FileExistsError) as exc_info:
            download(self.test_dataset_name, str(output_path))

        assert "already exists" in str(exc_info.value)
        assert "overwrite=True" in str(exc_info.value)
        assert "manually remove" in str(exc_info.value)

    @patch("nvflare.app_opt.kaggle.download_data.kagglehub")
    def test_download_existing_directory_with_overwrite_and_marker(self, mock_kagglehub):
        """Test downloading with overwrite=True when directory has marker (should succeed)"""
        # Setup existing directory with marker
        output_path = Path(self.temp_dir) / "output"
        output_path.mkdir()
        (output_path / ".kaggle_download_marker").touch()
        (output_path / "old_file.csv").touch()

        mock_download_path = Path(self.temp_dir) / "mock_download"
        mock_download_path.mkdir()
        (mock_download_path / "new_file.csv").touch()
        mock_kagglehub.dataset_download.return_value = str(mock_download_path)

        # Execute
        download(self.test_dataset_name, str(output_path), overwrite=True)

        # Verify
        assert output_path.exists()
        assert output_path.is_dir()
        assert (output_path / ".kaggle_download_marker").exists()
        assert (output_path / "new_file.csv").exists()
        assert not (output_path / "old_file.csv").exists()  # Old content removed

    @patch("nvflare.app_opt.kaggle.download_data.kagglehub")
    def test_download_existing_directory_without_marker(self, mock_kagglehub):
        """Test downloading when directory exists without marker (should fail even with overwrite)"""
        # Setup existing directory WITHOUT marker
        output_path = Path(self.temp_dir) / "output"
        output_path.mkdir()
        (output_path / "important_file.txt").touch()  # User's important file

        mock_download_path = Path(self.temp_dir) / "mock_download"
        mock_download_path.mkdir()
        mock_kagglehub.dataset_download.return_value = str(mock_download_path)

        # Execute and verify - should fail to protect user's data
        with pytest.raises(ValueError) as exc_info:
            download(self.test_dataset_name, str(output_path), overwrite=True)

        assert "doesn't appear to be a previous" in str(exc_info.value)
        assert "kaggle download" in str(exc_info.value)
        # Verify original file still exists (not removed)
        assert (output_path / "important_file.txt").exists()

    @patch("nvflare.app_opt.kaggle.download_data.kagglehub")
    def test_download_existing_file_not_directory(self, mock_kagglehub):
        """Test downloading when path exists as a file, not directory (should fail)"""
        # Setup - create a file instead of directory
        output_path = Path(self.temp_dir) / "output"
        output_path.touch()  # Create as file

        mock_download_path = Path(self.temp_dir) / "mock_download"
        mock_download_path.mkdir()
        mock_kagglehub.dataset_download.return_value = str(mock_download_path)

        # Execute and verify
        with pytest.raises(ValueError) as exc_info:
            download(self.test_dataset_name, str(output_path), overwrite=True)

        assert "not a directory" in str(exc_info.value)

    @patch("nvflare.app_opt.kaggle.download_data.kagglehub")
    def test_download_kagglehub_called_correctly(self, mock_kagglehub):
        """Test that kagglehub.dataset_download is called with correct parameters"""
        # Setup
        mock_download_path = Path(self.temp_dir) / "mock_download"
        mock_download_path.mkdir()
        mock_kagglehub.dataset_download.return_value = str(mock_download_path)

        output_path = Path(self.temp_dir) / "output"

        # Execute
        download(self.test_dataset_name, str(output_path))

        # Verify kagglehub was called correctly
        mock_kagglehub.dataset_download.assert_called_once_with(self.test_dataset_name)

    @patch("nvflare.app_opt.kaggle.download_data.kagglehub")
    def test_marker_file_created_after_successful_download(self, mock_kagglehub):
        """Test that marker file is created after successful download"""
        # Setup
        mock_download_path = Path(self.temp_dir) / "mock_download"
        mock_download_path.mkdir()
        mock_kagglehub.dataset_download.return_value = str(mock_download_path)

        output_path = Path(self.temp_dir) / "output"

        # Execute
        download(self.test_dataset_name, str(output_path))

        # Verify marker file exists
        marker_file = output_path / ".kaggle_download_marker"
        assert marker_file.exists()
        assert marker_file.is_file()

    @patch("nvflare.app_opt.kaggle.download_data.kagglehub")
    def test_download_overwrites_marker_file(self, mock_kagglehub):
        """Test that downloading again with overwrite refreshes the marker file"""
        # Setup - first download
        output_path = Path(self.temp_dir) / "output"
        output_path.mkdir()
        marker_file = output_path / ".kaggle_download_marker"
        marker_file.touch()

        # Modify marker file timestamp
        import os
        import time

        old_time = time.time() - 1000  # 1000 seconds ago
        os.utime(marker_file, (old_time, old_time))
        old_mtime = marker_file.stat().st_mtime

        mock_download_path = Path(self.temp_dir) / "mock_download"
        mock_download_path.mkdir()
        mock_kagglehub.dataset_download.return_value = str(mock_download_path)

        # Execute - second download with overwrite
        download(self.test_dataset_name, str(output_path), overwrite=True)

        # Verify marker file was refreshed (newer timestamp)
        new_mtime = marker_file.stat().st_mtime
        assert new_mtime > old_mtime

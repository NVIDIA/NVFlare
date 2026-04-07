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

"""Tests for push_folder key-load guard in FileTransferModule."""

import os
from unittest.mock import MagicMock, patch

from nvflare.fuel.hci.reg import CommandEntry


def _make_cmd_entry(scope_name="admin", cmd_name="push_folder"):
    """Create a real CommandEntry with a minimal scope mock."""
    scope = MagicMock()
    scope.name = scope_name
    return CommandEntry(
        scope=scope,
        name=cmd_name,
        desc="",
        usage="push_folder <folder>",
        handler=None,
        authz_func=None,
        visible=True,
        confirm=None,
        client_cmd=None,
    )


def _make_push_folder_args_and_ctx(key_path, cert_path, folder_name="test_job"):
    """Build mock args and CommandContext for push_folder."""
    args = [None, folder_name]

    api = MagicMock()
    api.client_key = key_path
    api.client_cert = cert_path

    ctx = MagicMock()
    ctx.get_command_entry.return_value = _make_cmd_entry()
    ctx.get_api.return_value = api

    return args, ctx


class TestPushFolderKeyGuard:
    """Test that load_private_key_file + sign_folders are skipped when key absent."""

    def _run_push_folder(self, tmp_path, key_path, cert_path):
        """Helper: create upload dir structure and call push_folder."""
        from nvflare.fuel.hci.client.file_transfer import FileTransferModule

        upload_dir = str(tmp_path / "upload")
        download_dir = str(tmp_path / "dl")
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(download_dir, exist_ok=True)

        folder_name = "test_job"
        os.makedirs(os.path.join(upload_dir, folder_name), exist_ok=True)

        module = FileTransferModule(upload_dir=upload_dir, download_dir=download_dir)
        args, ctx = _make_push_folder_args_and_ctx(key_path, cert_path, folder_name)

        with (
            patch("nvflare.fuel.hci.client.file_transfer.load_private_key_file") as mock_load,
            patch("nvflare.fuel.hci.client.file_transfer.sign_folders") as mock_sign,
            patch("nvflare.fuel.hci.client.file_transfer.zip_directory_to_file"),
            patch.object(ctx.get_api.return_value, "server_execute", return_value={}),
        ):
            module.push_folder(args, ctx)
            return mock_load, mock_sign

    def test_key_none_skips_signing(self, tmp_path):
        """key_path=None: neither load nor sign called."""
        mock_load, mock_sign = self._run_push_folder(tmp_path, None, None)
        mock_load.assert_not_called()
        mock_sign.assert_not_called()

    def test_key_empty_string_skips_signing(self, tmp_path):
        """key_path='': neither load nor sign called."""
        mock_load, mock_sign = self._run_push_folder(tmp_path, "", None)
        mock_load.assert_not_called()
        mock_sign.assert_not_called()

    def test_key_path_set_but_file_missing_skips_signing(self, tmp_path):
        """key_path points to nonexistent file: guard clause 2 fails, skip."""
        missing_key = str(tmp_path / "nonexistent.key")
        mock_load, mock_sign = self._run_push_folder(tmp_path, missing_key, "cert.crt")
        mock_load.assert_not_called()
        mock_sign.assert_not_called()

    def test_key_present_cert_none_skips_signing(self, tmp_path):
        """Key file exists but cert is None: guard clause 3 fails, skip."""
        key_file = tmp_path / "test.key"
        key_file.write_text("fake key content")
        mock_load, mock_sign = self._run_push_folder(tmp_path, str(key_file), None)
        mock_load.assert_not_called()
        mock_sign.assert_not_called()

    def test_key_present_cert_empty_skips_signing(self, tmp_path):
        """Key file exists but cert is empty string: guard clause 3 fails, skip."""
        key_file = tmp_path / "test.key"
        key_file.write_text("fake key content")
        mock_load, mock_sign = self._run_push_folder(tmp_path, str(key_file), "")
        mock_load.assert_not_called()
        mock_sign.assert_not_called()

    def test_both_present_signing_occurs(self, tmp_path):
        """Key and cert both present: load and sign are called."""
        key_file = tmp_path / "test.key"
        key_file.write_text("fake key content")

        upload_dir = str(tmp_path / "upload")
        download_dir = str(tmp_path / "dl")
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(download_dir, exist_ok=True)

        folder_name = "test_job"
        os.makedirs(os.path.join(upload_dir, folder_name), exist_ok=True)

        from nvflare.fuel.hci.client.file_transfer import FileTransferModule

        module = FileTransferModule(upload_dir=upload_dir, download_dir=download_dir)
        args, ctx = _make_push_folder_args_and_ctx(str(key_file), "/path/to/cert.crt", folder_name)

        fake_key = MagicMock()
        with (
            patch("nvflare.fuel.hci.client.file_transfer.load_private_key_file", return_value=fake_key) as mock_load,
            patch("nvflare.fuel.hci.client.file_transfer.sign_folders") as mock_sign,
            patch("nvflare.fuel.hci.client.file_transfer.zip_directory_to_file"),
            patch.object(ctx.get_api.return_value, "server_execute", return_value={}),
        ):
            module.push_folder(args, ctx)
            mock_load.assert_called_once_with(str(key_file))
            mock_sign.assert_called_once()

    def test_sign_folders_exception_returns_error(self, tmp_path):
        """sign_folders raising an exception returns ERROR_RUNTIME instead of crashing."""
        from nvflare.fuel.hci.client.api_status import APIStatus
        from nvflare.fuel.hci.client.file_transfer import FileTransferModule

        key_file = tmp_path / "test.key"
        key_file.write_text("fake key content")

        upload_dir = str(tmp_path / "upload")
        download_dir = str(tmp_path / "dl")
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(download_dir, exist_ok=True)

        folder_name = "test_job"
        os.makedirs(os.path.join(upload_dir, folder_name), exist_ok=True)

        module = FileTransferModule(upload_dir=upload_dir, download_dir=download_dir)
        args, ctx = _make_push_folder_args_and_ctx(str(key_file), "/path/to/cert.crt", folder_name)

        with (
            patch(
                "nvflare.fuel.hci.client.file_transfer.load_private_key_file",
                side_effect=ValueError("corrupted key"),
            ),
        ):
            result = module.push_folder(args, ctx)

        assert result["status"] == APIStatus.ERROR_RUNTIME
        assert "corrupted key" in result["details"]

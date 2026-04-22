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
import zipfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.app_opt.job_launcher.workspace_cell_transfer import (
    BOOTSTRAP_CONNECT_TIMEOUT,
    ENV_WORKSPACE_OWNER_FQCN,
    ENV_WORKSPACE_TRANSFER_TOKEN,
    WorkspaceTransferManager,
    _hash_file,
    _wait_for_bootstrap_ready,
    download_workspace,
    make_workspace_transfer_fqcn,
    upload_results,
)
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply, new_cell_message

JOB_ID = "abc12345-dead-beef-0000-111122223333"


def _write_file(path: str, content: bytes = b"data") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(content)


def _make_workspace(root: str, job_id: str) -> None:
    _write_file(os.path.join(root, "local", "resources.json"), b'{"resources":{}}')
    _write_file(os.path.join(root, job_id, "app", "config", "config_train.json"), b'{"rounds":3}')


def _make_zip(path: str, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries.items():
            zf.writestr(name, content)


class _FakeCell:
    def __init__(self, fqcn="owner.cell", reply=None, backbone_ready=True, connected=True):
        self._fqcn = fqcn
        self.reply = reply
        self.callbacks = {}
        self.requests = []
        self.backbone_ready = backbone_ready
        self.connected = connected

    def get_fqcn(self):
        return self._fqcn

    def register_request_cb(self, channel, topic, cb):
        self.callbacks[(channel, topic)] = cb

    def send_request(self, **kwargs):
        self.requests.append(kwargs)
        if callable(self.reply):
            return self.reply(kwargs)
        return self.reply

    def is_backbone_ready(self):
        if callable(self.backbone_ready):
            return self.backbone_ready()
        return self.backbone_ready

    def is_cell_connected(self, _target_fqcn):
        if callable(self.connected):
            return self.connected()
        return self.connected

    def stop(self):
        pass


class TestGetOrCreate:
    def test_returns_same_manager_for_same_cell(self):
        owner_cell = _FakeCell(fqcn="site-1.parent")
        first = WorkspaceTransferManager.get_or_create(owner_cell)
        second = WorkspaceTransferManager.get_or_create(owner_cell)
        try:
            assert first is second
        finally:
            # handlers are registered once per (channel, topic); same manager means one pair
            assert len(owner_cell.callbacks) == 2


class TestWorkspaceTransferManager:
    def test_prepare_download_returns_ref_for_valid_token(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root:
            _make_workspace(ws_root, JOB_ID)
            owner_cell = _FakeCell(fqcn="site-1.parent")
            manager = WorkspaceTransferManager(owner_cell)
            transfer_token = manager.add_job(JOB_ID, ws_root)

            fake_downloader = MagicMock()
            fake_downloader.tx_id = "tx-1"
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer.ObjectDownloader",
                lambda *args, **kwargs: fake_downloader,
            )
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer.add_file",
                lambda downloader, file_name: "ref-1",
            )

            request = new_cell_message({}, {"job_id": JOB_ID, "transfer_token": transfer_token})
            request.set_header(MessageHeaderKey.ORIGIN, make_workspace_transfer_fqcn(owner_cell.get_fqcn(), JOB_ID))
            reply = manager._handle_prepare_download(request)
            try:
                assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
                assert reply.payload["ref_id"] == "ref-1"
                bundle_path = manager.jobs[JOB_ID].download_bundle_path
                assert os.path.exists(bundle_path)
                assert reply.payload["sha256"] == _hash_file(bundle_path)
            finally:
                manager.remove_job(JOB_ID)

    def test_prepare_download_rejects_wrong_token(self):
        with tempfile.TemporaryDirectory() as ws_root:
            owner_cell = _FakeCell(fqcn="site-1.parent")
            manager = WorkspaceTransferManager(owner_cell)
            manager.add_job(JOB_ID, ws_root)
            request = new_cell_message({}, {"job_id": JOB_ID, "transfer_token": "wrong-token"})
            request.set_header(MessageHeaderKey.ORIGIN, make_workspace_transfer_fqcn(owner_cell.get_fqcn(), JOB_ID))
            try:
                reply = manager._handle_prepare_download(request)
                assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED
            finally:
                manager.remove_job(JOB_ID)

    def test_prepare_download_allows_missing_origin_when_token_matches(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root:
            _make_workspace(ws_root, JOB_ID)
            owner_cell = _FakeCell(fqcn="site-1.parent")
            manager = WorkspaceTransferManager(owner_cell)
            transfer_token = manager.add_job(JOB_ID, ws_root)

            fake_downloader = MagicMock()
            fake_downloader.tx_id = "tx-1"
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer.ObjectDownloader",
                lambda *args, **kwargs: fake_downloader,
            )
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer.add_file",
                lambda downloader, file_name: "ref-1",
            )

            request = new_cell_message({}, {"job_id": JOB_ID, "transfer_token": transfer_token})
            try:
                reply = manager._handle_prepare_download(request)
                assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
            finally:
                manager.remove_job(JOB_ID)

    def test_prepare_download_returns_error_when_bundle_creation_fails(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root:
            owner_cell = _FakeCell(fqcn="site-1.parent")
            manager = WorkspaceTransferManager(owner_cell)
            transfer_token = manager.add_job(JOB_ID, ws_root)

            def _boom(*_args, **_kwargs):
                raise OSError("disk full")

            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer._zip_workspace_to_file",
                _boom,
            )

            request = new_cell_message({}, {"job_id": JOB_ID, "transfer_token": transfer_token})
            request.set_header(MessageHeaderKey.ORIGIN, make_workspace_transfer_fqcn(owner_cell.get_fqcn(), JOB_ID))
            try:
                reply = manager._handle_prepare_download(request)
                assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
                assert "failed to prepare workspace download" in reply.get_header(MessageHeaderKey.ERROR)
            finally:
                manager.remove_job(JOB_ID)

    def test_publish_results_extracts_job_dir_only(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as tmp:
            owner_cell = _FakeCell(fqcn="site-1.parent")
            manager = WorkspaceTransferManager(owner_cell)
            transfer_token = manager.add_job(JOB_ID, ws_root)

            zip_path = os.path.join(tmp, "results.zip")
            _make_zip(zip_path, {f"{JOB_ID}/result.txt": b"done"})
            zip_sha = _hash_file(zip_path)

            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer.download_file",
                lambda **kwargs: (None, zip_path),
            )

            request = new_cell_message(
                {},
                {"job_id": JOB_ID, "ref_id": "ref-1", "transfer_token": transfer_token, "sha256": zip_sha},
            )
            request.set_header(MessageHeaderKey.ORIGIN, make_workspace_transfer_fqcn(owner_cell.get_fqcn(), JOB_ID))
            try:
                reply = manager._handle_publish_results(request)
                assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
                with open(os.path.join(ws_root, JOB_ID, "result.txt"), "rb") as fh:
                    assert fh.read() == b"done"
                assert JOB_ID not in manager.jobs
            finally:
                manager.remove_job(JOB_ID)

    def test_publish_results_rejects_other_job_dir(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as tmp:
            owner_cell = _FakeCell(fqcn="site-1.parent")
            manager = WorkspaceTransferManager(owner_cell)
            transfer_token = manager.add_job(JOB_ID, ws_root)

            zip_path = os.path.join(tmp, "results.zip")
            _make_zip(zip_path, {"other-job/result.txt": b"oops"})
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer.download_file",
                lambda **kwargs: (None, zip_path),
            )

            request = new_cell_message(
                {},
                {"job_id": JOB_ID, "ref_id": "ref-1", "transfer_token": transfer_token},
            )
            request.set_header(MessageHeaderKey.ORIGIN, make_workspace_transfer_fqcn(owner_cell.get_fqcn(), JOB_ID))
            try:
                reply = manager._handle_publish_results(request)
                assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
                assert not os.path.exists(os.path.join(ws_root, "other-job", "result.txt"))
            finally:
                manager.remove_job(JOB_ID)

    def test_publish_results_rejects_wrong_token(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as tmp:
            owner_cell = _FakeCell(fqcn="site-1.parent")
            manager = WorkspaceTransferManager(owner_cell)
            manager.add_job(JOB_ID, ws_root)

            zip_path = os.path.join(tmp, "results.zip")
            _make_zip(zip_path, {f"{JOB_ID}/result.txt": b"done"})
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer.download_file",
                lambda **kwargs: (None, zip_path),
            )

            request = new_cell_message(
                {},
                {"job_id": JOB_ID, "ref_id": "ref-1", "transfer_token": "wrong-token"},
            )
            request.set_header(MessageHeaderKey.ORIGIN, make_workspace_transfer_fqcn(owner_cell.get_fqcn(), JOB_ID))
            try:
                reply = manager._handle_publish_results(request)
                assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED
            finally:
                manager.remove_job(JOB_ID)

    def test_publish_results_requires_origin_for_download(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as tmp:
            owner_cell = _FakeCell(fqcn="site-1.parent")
            manager = WorkspaceTransferManager(owner_cell)
            transfer_token = manager.add_job(JOB_ID, ws_root)

            zip_path = os.path.join(tmp, "results.zip")
            _make_zip(zip_path, {f"{JOB_ID}/result.txt": b"done"})
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer.download_file",
                lambda **kwargs: (None, zip_path),
            )

            request = new_cell_message(
                {},
                {"job_id": JOB_ID, "ref_id": "ref-1", "transfer_token": transfer_token},
            )
            try:
                reply = manager._handle_publish_results(request)
                assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
            finally:
                manager.remove_job(JOB_ID)

    def test_prepare_download_rejects_missing_token(self):
        with tempfile.TemporaryDirectory() as ws_root:
            owner_cell = _FakeCell(fqcn="site-1.parent")
            manager = WorkspaceTransferManager(owner_cell)
            manager.add_job(JOB_ID, ws_root)
            request = new_cell_message({}, {"job_id": JOB_ID})
            request.set_header(MessageHeaderKey.ORIGIN, make_workspace_transfer_fqcn(owner_cell.get_fqcn(), JOB_ID))
            try:
                reply = manager._handle_prepare_download(request)
                assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
            finally:
                manager.remove_job(JOB_ID)

    def test_publish_results_rejects_missing_token(self):
        with tempfile.TemporaryDirectory() as ws_root:
            owner_cell = _FakeCell(fqcn="site-1.parent")
            manager = WorkspaceTransferManager(owner_cell)
            manager.add_job(JOB_ID, ws_root)
            request = new_cell_message({}, {"job_id": JOB_ID, "ref_id": "ref-1"})
            request.set_header(MessageHeaderKey.ORIGIN, make_workspace_transfer_fqcn(owner_cell.get_fqcn(), JOB_ID))
            try:
                reply = manager._handle_publish_results(request)
                assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
            finally:
                manager.remove_job(JOB_ID)


class TestWorkspaceBootstrapHelpers:
    def test_wait_for_bootstrap_ready_succeeds_after_connection(self, monkeypatch):
        readiness = iter([False, False, True])
        fake_cell = _FakeCell(
            fqcn="site-1.parent",
            backbone_ready=lambda: next(readiness),
            connected=True,
        )
        monkeypatch.setattr("nvflare.app_opt.job_launcher.workspace_cell_transfer.time.sleep", lambda _: None)
        _wait_for_bootstrap_ready(fake_cell, "site-1.parent", timeout=BOOTSTRAP_CONNECT_TIMEOUT)

    def test_wait_for_bootstrap_ready_times_out(self, monkeypatch):
        fake_cell = _FakeCell(fqcn="site-1.parent", backbone_ready=False, connected=False)
        ticks = iter([0.0, 0.05, 0.11])
        monkeypatch.setattr("nvflare.app_opt.job_launcher.workspace_cell_transfer.time.sleep", lambda _: None)
        monkeypatch.setattr(
            "nvflare.app_opt.job_launcher.workspace_cell_transfer.time.monotonic",
            lambda: next(ticks),
        )
        with pytest.raises(RuntimeError, match="failed to connect to parent"):
            _wait_for_bootstrap_ready(fake_cell, "site-1.parent", timeout=0.1)

    def test_download_workspace_noop_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv(ENV_WORKSPACE_OWNER_FQCN, raising=False)
        monkeypatch.delenv(ENV_WORKSPACE_TRANSFER_TOKEN, raising=False)
        args = SimpleNamespace(workspace="/tmp/workspace", job_id=JOB_ID, parent_url="tcp://parent")
        download_workspace(args, secure_mode=False)

    def test_download_workspace_raises_when_token_missing(self, monkeypatch):
        monkeypatch.setenv(ENV_WORKSPACE_OWNER_FQCN, "site-1.parent")
        monkeypatch.delenv(ENV_WORKSPACE_TRANSFER_TOKEN, raising=False)
        args = SimpleNamespace(workspace="/tmp/workspace", job_id=JOB_ID, parent_url="tcp://parent")
        with pytest.raises(RuntimeError, match=ENV_WORKSPACE_TRANSFER_TOKEN):
            download_workspace(args, secure_mode=False)

    def test_download_workspace_extracts_bundle(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as tmp:
            zip_path = os.path.join(tmp, "workspace.zip")
            _make_zip(
                zip_path,
                {
                    "local/resources.json": b'{"resources":{}}',
                    f"{JOB_ID}/app/config/config_train.json": b'{"rounds":3}',
                },
            )
            zip_sha = _hash_file(zip_path)
            fake_cell = _FakeCell(
                fqcn="site-1.parent",
                reply=make_reply(ReturnCode.OK, body={"job_id": JOB_ID, "ref_id": "ref-1", "sha256": zip_sha}),
            )
            monkeypatch.setenv(ENV_WORKSPACE_OWNER_FQCN, "site-1.parent")
            monkeypatch.setenv(ENV_WORKSPACE_TRANSFER_TOKEN, "token-1")
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer._get_bootstrap_cell",
                lambda *a, **kw: fake_cell,
            )
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer._close_bootstrap_cell",
                lambda: None,
            )
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer.download_file",
                lambda **kwargs: (None, zip_path),
            )
            args = SimpleNamespace(
                workspace=ws_root,
                job_id=JOB_ID,
                parent_url="tcp://parent",
                root_url="tcp://root",
            )

            download_workspace(args, secure_mode=False)

            request = fake_cell.requests[0]["request"]
            assert request.payload["transfer_token"] == "token-1"
            assert os.path.exists(os.path.join(ws_root, "local", "resources.json"))
            assert os.path.exists(os.path.join(ws_root, JOB_ID, "app", "config", "config_train.json"))

    def test_upload_results_noop_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv(ENV_WORKSPACE_OWNER_FQCN, raising=False)
        monkeypatch.delenv(ENV_WORKSPACE_TRANSFER_TOKEN, raising=False)
        args = SimpleNamespace(workspace="/tmp/workspace", job_id=JOB_ID, parent_url="tcp://parent")
        upload_results(args, secure_mode=False)

    def test_upload_results_raises_when_token_missing(self, monkeypatch):
        monkeypatch.setenv(ENV_WORKSPACE_OWNER_FQCN, "site-1.parent")
        monkeypatch.delenv(ENV_WORKSPACE_TRANSFER_TOKEN, raising=False)
        args = SimpleNamespace(workspace="/tmp/workspace", job_id=JOB_ID, parent_url="tcp://parent")
        with pytest.raises(RuntimeError, match=ENV_WORKSPACE_TRANSFER_TOKEN):
            upload_results(args, secure_mode=False)

    def test_upload_results_publishes_ref_to_parent(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root:
            _write_file(os.path.join(ws_root, JOB_ID, "result.txt"), b"done")
            fake_cell = _FakeCell(fqcn="site-1.parent", reply=make_reply(ReturnCode.OK, body={"job_id": JOB_ID}))
            fake_downloader = MagicMock()
            fake_downloader.tx_id = "tx-upload"

            monkeypatch.setenv(ENV_WORKSPACE_OWNER_FQCN, "site-1.parent")
            monkeypatch.setenv(ENV_WORKSPACE_TRANSFER_TOKEN, "token-1")
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer._get_bootstrap_cell",
                lambda *a, **kw: fake_cell,
            )
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer._close_bootstrap_cell",
                lambda: None,
            )
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer.ObjectDownloader",
                lambda *args, **kwargs: fake_downloader,
            )
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer.add_file",
                lambda downloader, file_name: "ref-upload",
            )

            args = SimpleNamespace(
                workspace=ws_root,
                job_id=JOB_ID,
                parent_url="tcp://parent",
                root_url="tcp://root",
            )

            upload_results(args, secure_mode=False)

            request = fake_cell.requests[0]["request"]
            assert request.payload["job_id"] == JOB_ID
            assert request.payload["ref_id"] == "ref-upload"
            assert request.payload["transfer_token"] == "token-1"
            fake_downloader.delete_transaction.assert_called_once_with()

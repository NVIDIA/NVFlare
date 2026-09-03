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

import json
import os
import stat
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
    _bootstrap_auth_identity_map,
    _create_bootstrap_cell,
    _get_bootstrap_tls_pair,
    _hash_file,
    _install_job_cert,
    _wait_for_bootstrap_ready,
    _zip_results_to_file,
    _zip_workspace_to_file,
    download_workspace,
    make_workspace_transfer_fqcn,
    upload_results,
    upload_results_on_shutdown,
)
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.utils import make_reply, new_cell_message
from nvflare.fuel.f3.drivers.driver_params import DriverParams

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


def _make_zip_with_symlink(path: str, link_name: str, link_target: str, file_entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo(link_name)
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, link_target)
        for name, content in file_entries.items():
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
    def test_workspace_bundle_excludes_internal_study_config_files(self):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as tmp:
            _make_workspace(ws_root, JOB_ID)
            _write_file(
                os.path.join(ws_root, "local", "study_data.yaml"),
                b"study-a:\n  training:\n    source: nvfldata\n    mode: ro\n",
            )
            _write_file(os.path.join(ws_root, "local", "study_runtime.yaml"), b"format_version: 2\nstudies: {}\n")
            _write_file(os.path.join(ws_root, "local", "custom", "helper.py"), b"VALUE = 1\n")
            zip_path = os.path.join(tmp, "workspace.zip")

            _zip_workspace_to_file(ws_root, JOB_ID, zip_path)

            with zipfile.ZipFile(zip_path) as zf:
                names = set(zf.namelist())
            assert "local/resources.json" in names
            assert "local/custom/helper.py" in names
            assert f"{JOB_ID}/app/config/config_train.json" in names
            assert "local/study_data.yaml" not in names
            assert "local/study_runtime.yaml" not in names

    def test_workspace_bundle_excludes_legacy_custom_study_data_path(self):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as tmp:
            _make_workspace(ws_root, JOB_ID)
            resource_config = {
                "components": [
                    {
                        "id": "k8s_launcher",
                        "path": "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher",
                        "args": {
                            "workspace_mount_path": "/workspace",
                            "study_data_pvc_file_path": "/workspace/local/custom_data.yaml",
                        },
                    }
                ]
            }
            _write_file(os.path.join(ws_root, "local", "resources.json"), json.dumps(resource_config).encode())
            _write_file(os.path.join(ws_root, "local", "custom_data.yaml"), b"study-a: {}\n")
            _write_file(os.path.join(ws_root, "local", "custom", "helper.py"), b"VALUE = 1\n")
            zip_path = os.path.join(tmp, "workspace.zip")

            _zip_workspace_to_file(ws_root, JOB_ID, zip_path)

            with zipfile.ZipFile(zip_path) as zf:
                names = set(zf.namelist())
            assert "local/resources.json" in names
            assert "local/custom/helper.py" in names
            assert "local/custom_data.yaml" not in names

    def test_workspace_bundle_excludes_study_runtime_pod_templates(self):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as tmp:
            _make_workspace(ws_root, JOB_ID)
            _write_file(
                os.path.join(ws_root, "local", "study_runtime.yaml"),
                b"format_version: 2\n"
                b"studies:\n"
                b"  study-a:\n"
                b"    pod_template: pod_specs/h100-pod.yaml\n"
                b"  study-b:\n"
                b"    pod_template:\n"
                b"      spec: {}\n",
            )
            _write_file(os.path.join(ws_root, "local", "pod_specs", "h100-pod.yaml"), b"kind: Pod\n")
            _write_file(os.path.join(ws_root, "local", "custom", "helper.py"), b"VALUE = 1\n")
            zip_path = os.path.join(tmp, "workspace.zip")

            _zip_workspace_to_file(ws_root, JOB_ID, zip_path)

            with zipfile.ZipFile(zip_path) as zf:
                names = set(zf.namelist())
            assert "local/resources.json" in names
            assert "local/custom/helper.py" in names
            assert f"{JOB_ID}/app/config/config_train.json" in names
            assert "local/study_runtime.yaml" not in names
            assert "local/pod_specs/h100-pod.yaml" not in names

    @pytest.mark.parametrize("zip_fn", [_zip_workspace_to_file, _zip_results_to_file])
    def test_bundles_exclude_job_credential(self, zip_fn):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as tmp:
            _make_workspace(ws_root, JOB_ID)
            _write_file(os.path.join(ws_root, JOB_ID, "job_cert", "job.crt"), b"cert")
            _write_file(os.path.join(ws_root, JOB_ID, "job_cert", "job.key"), b"key")
            zip_path = os.path.join(tmp, "bundle.zip")

            zip_fn(ws_root, JOB_ID, zip_path)

            with zipfile.ZipFile(zip_path) as zf:
                names = set(zf.namelist())
            assert f"{JOB_ID}/app/config/config_train.json" in names
            assert not any(name.startswith(f"{JOB_ID}/job_cert/") for name in names)

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

    def test_publish_results_rejects_symlink_entries(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as tmp:
            owner_cell = _FakeCell(fqcn="site-1.parent")
            manager = WorkspaceTransferManager(owner_cell)
            transfer_token = manager.add_job(JOB_ID, ws_root)

            zip_path = os.path.join(tmp, "results-symlink.zip")
            _make_zip_with_symlink(
                zip_path,
                f"{JOB_ID}/link",
                "../../startup",
                {f"{JOB_ID}/link/result.txt": b"oops"},
            )
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
                assert "symlink not allowed" in reply.get_header(MessageHeaderKey.ERROR)
            finally:
                manager.remove_job(JOB_ID)

    def test_publish_results_returns_error_on_unexpected_os_error(self, monkeypatch):
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
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer.os.makedirs",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
            )

            request = new_cell_message(
                {},
                {"job_id": JOB_ID, "ref_id": "ref-1", "transfer_token": transfer_token},
            )
            request.set_header(MessageHeaderKey.ORIGIN, make_workspace_transfer_fqcn(owner_cell.get_fqcn(), JOB_ID))
            try:
                reply = manager._handle_publish_results(request)
                assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
                assert "unexpected error processing results" in reply.get_header(MessageHeaderKey.ERROR)
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

    def test_upload_results_raises_when_run_dir_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ENV_WORKSPACE_OWNER_FQCN, "site-1.parent")
        monkeypatch.setenv(ENV_WORKSPACE_TRANSFER_TOKEN, "token-1")
        args = SimpleNamespace(workspace=str(tmp_path), job_id=JOB_ID, parent_url="tcp://parent")

        with pytest.raises(RuntimeError, match="results workspace does not exist"):
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

    def test_upload_results_raises_on_negative_reply(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root:
            _write_file(os.path.join(ws_root, JOB_ID, "result.txt"), b"done")
            fake_cell = _FakeCell(
                fqcn="site-1.parent",
                reply=make_reply(ReturnCode.COMM_ERROR, error="receiver failed"),
            )
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

            with pytest.raises(RuntimeError, match="results upload failed"):
                upload_results(args, secure_mode=False)

            fake_downloader.delete_transaction.assert_called_once_with()

    def test_upload_results_waits_for_bootstrap_ready(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root:
            _write_file(os.path.join(ws_root, JOB_ID, "result.txt"), b"done")
            fake_cell = _FakeCell(fqcn="site-1.parent", reply=make_reply(ReturnCode.OK, body={"job_id": JOB_ID}))
            fake_downloader = MagicMock()
            wait_calls = []

            monkeypatch.setenv(ENV_WORKSPACE_OWNER_FQCN, "site-1.parent")
            monkeypatch.setenv(ENV_WORKSPACE_TRANSFER_TOKEN, "token-1")
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer._get_bootstrap_cell",
                lambda *a, **kw: fake_cell,
            )
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer._wait_for_bootstrap_ready",
                lambda cell, owner_fqcn: wait_calls.append((cell, owner_fqcn)),
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

            assert wait_calls == [(fake_cell, "site-1.parent")]

    def test_upload_results_cleans_temp_bundle_when_zip_creation_fails(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws_root, tempfile.TemporaryDirectory() as tmp:
            _write_file(os.path.join(ws_root, JOB_ID, "result.txt"), b"done")
            created = {}

            real_named_temporary_file = tempfile.NamedTemporaryFile

            def _named_tmp(*args, **kwargs):
                tmp_file = real_named_temporary_file(*args, dir=tmp, **kwargs)
                created["path"] = tmp_file.name
                return tmp_file

            monkeypatch.setenv(ENV_WORKSPACE_OWNER_FQCN, "site-1.parent")
            monkeypatch.setenv(ENV_WORKSPACE_TRANSFER_TOKEN, "token-1")
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer.tempfile.NamedTemporaryFile",
                _named_tmp,
            )
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer._zip_results_to_file",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
            )
            monkeypatch.setattr(
                "nvflare.app_opt.job_launcher.workspace_cell_transfer._close_bootstrap_cell",
                lambda: None,
            )

            args = SimpleNamespace(
                workspace=ws_root,
                job_id=JOB_ID,
                parent_url="tcp://parent",
                root_url="tcp://root",
            )

            with pytest.raises(OSError, match="disk full"):
                upload_results(args, secure_mode=False)

            assert created["path"]
            assert not os.path.exists(created["path"])

    def test_upload_results_on_shutdown_propagates_publication_failure(self, monkeypatch):
        args = SimpleNamespace(job_id=JOB_ID)
        monkeypatch.setattr(
            "nvflare.app_opt.job_launcher.workspace_cell_transfer.upload_results",
            MagicMock(side_effect=RuntimeError("publication failed")),
        )

        with pytest.raises(RuntimeError, match="publication failed"):
            upload_results_on_shutdown(args, secure_mode=False)

    def test_upload_results_on_shutdown_preserves_primary_failure(self, monkeypatch):
        args = SimpleNamespace(job_id=JOB_ID)
        log = MagicMock()
        monkeypatch.setattr(
            "nvflare.app_opt.job_launcher.workspace_cell_transfer.upload_results",
            MagicMock(side_effect=RuntimeError("publication failed")),
        )

        with pytest.raises(RuntimeError, match="execution failed"):
            try:
                raise RuntimeError("execution failed")
            finally:
                upload_results_on_shutdown(args, secure_mode=False, log=log)

        log.error.assert_called_once()


class TestBootstrapAuthIdentityMap:
    def test_maps_logical_root_to_fed_client_server_identity(self, tmp_path):
        startup = tmp_path / "startup"
        startup.mkdir()
        (startup / "fed_client.json").write_text(
            json.dumps(
                {
                    "servers": [{"name": "project", "identity": "gcp-server"}],
                    "client": {"auth_identity_map": {"relay-a": "relay-a-cn"}},
                }
            )
        )

        identity_map = _bootstrap_auth_identity_map(str(startup))

        assert identity_map[FQCN.ROOT_SERVER] == "gcp-server"
        assert identity_map["relay-a"] == "relay-a-cn"

    def test_prefers_auth_identity_over_identity(self, tmp_path):
        startup = tmp_path / "startup"
        startup.mkdir()
        (startup / "fed_client.json").write_text(
            json.dumps(
                {
                    "servers": [
                        {
                            "name": "project",
                            "identity": "server",
                            "auth_identity": "gcp-server",
                        }
                    ]
                }
            )
        )

        identity_map = _bootstrap_auth_identity_map(str(startup))

        assert identity_map == {FQCN.ROOT_SERVER: "gcp-server"}

    def test_returns_none_when_startup_has_no_server_identity(self, tmp_path):
        startup = tmp_path / "startup"
        startup.mkdir()

        assert _bootstrap_auth_identity_map(str(startup)) is None

    def test_create_bootstrap_cell_passes_identity_map(self, monkeypatch, tmp_path):
        startup = tmp_path / "startup"
        startup.mkdir()
        (startup / "rootCA.pem").write_text("ca")
        (startup / "fed_client.json").write_text(
            json.dumps({"servers": [{"name": "project", "identity": "gcp-server"}]})
        )
        _write_file(str(tmp_path / JOB_ID / "job_cert" / "job.crt"), b"job-cert")
        _write_file(str(tmp_path / JOB_ID / "job_cert" / "job.key"), b"job-key")

        captured = {}

        class _FakeCell:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def start(self):
                pass

        monkeypatch.setattr("nvflare.app_opt.job_launcher.workspace_cell_transfer.Cell", _FakeCell)
        monkeypatch.setattr("nvflare.app_opt.job_launcher.workspace_cell_transfer.NetAgent", lambda cell: MagicMock())
        monkeypatch.setattr(
            "nvflare.app_opt.job_launcher.workspace_cell_transfer.set_add_auth_headers_filters",
            lambda *args, **kwargs: None,
        )

        args = SimpleNamespace(
            workspace=str(tmp_path),
            job_id=JOB_ID,
            parent_url="tcp://parent",
            root_url="tcp://root",
            client_name="site-1",
            token="token",
            token_signature="sig",
            ssid="ssid",
        )

        _create_bootstrap_cell(args, "site-1", True)

        assert captured["auth_identity_map"] == {FQCN.ROOT_SERVER: "gcp-server"}
        assert captured["secure"] is True
        job_cert_dir = str(tmp_path / JOB_ID / "job_cert")
        assert captured["credentials"][DriverParams.CLIENT_CERT.value] == os.path.join(job_cert_dir, "job.crt")
        assert captured["credentials"][DriverParams.CLIENT_KEY.value] == os.path.join(job_cert_dir, "job.key")

    def test_bootstrap_tls_pair_uses_job_credential_in_the_peer_role(self, tmp_path):
        run_dir = tmp_path / JOB_ID
        job_crt = run_dir / "job_cert" / "job.crt"
        job_key = run_dir / "job_cert" / "job.key"
        _write_file(str(job_crt), b"job-cert")
        _write_file(str(job_key), b"job-key")

        cert_path, key_path, cert_key, key_key = _get_bootstrap_tls_pair(str(run_dir), "site-1")

        assert (cert_path, key_path) == (str(job_crt), str(job_key))
        assert (cert_key, key_key) == (DriverParams.CLIENT_CERT.value, DriverParams.CLIENT_KEY.value)

        _, _, cert_key, key_key = _get_bootstrap_tls_pair(str(run_dir), FQCN.ROOT_SERVER)
        assert (cert_key, key_key) == (DriverParams.SERVER_CERT.value, DriverParams.SERVER_KEY.value)

    def test_bootstrap_tls_pair_requires_job_credential(self, tmp_path):
        with pytest.raises(RuntimeError, match="requires the job credential"):
            _get_bootstrap_tls_pair(str(tmp_path / JOB_ID), "site-1")

    def test_install_job_cert_writes_run_dir(self, tmp_path):
        args = SimpleNamespace(workspace=str(tmp_path), job_id=JOB_ID, job_cert_pem="cert-pem", job_key_pem="key-pem")

        _install_job_cert(args)

        cert_path = tmp_path / JOB_ID / "job_cert" / "job.crt"
        key_path = tmp_path / JOB_ID / "job_cert" / "job.key"
        assert cert_path.read_bytes() == b"cert-pem"
        assert key_path.read_bytes() == b"key-pem"
        assert stat.S_IMODE(os.stat(key_path).st_mode) == 0o600

    def test_install_job_cert_noop_without_complete_credential(self, tmp_path):
        args = SimpleNamespace(workspace=str(tmp_path), job_id=JOB_ID, job_cert_pem=None, job_key_pem="key-only")

        _install_job_cert(args)

        assert not (tmp_path / JOB_ID).exists()

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
"""CellNet-based workspace transfer for launched jobs.

The parent process exposes a small transfer service on its existing CellNet cell.
Launched job pods create short-lived bootstrap child cells to:

1. request a workspace bundle from the parent
2. upload final job results back to the parent

The actual payload transfer uses the existing F3 file downloader infrastructure,
so large bundles move in chunks instead of being buffered into a single message.
"""
from __future__ import annotations

import hashlib
import logging
import os
import secrets
import shutil
import stat
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.cellnet.utils import make_reply, new_cell_message
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.download_service import DownloadService
from nvflare.fuel.f3.streaming.file_downloader import add_file, download_file
from nvflare.fuel.f3.streaming.obj_downloader import ObjectDownloader
from nvflare.fuel.sec.authn import set_add_auth_headers_filters
from nvflare.private.defs import AUTH_CLIENT_NAME_FOR_SJ
from nvflare.security.logging import secure_format_exception

logger = logging.getLogger(__name__)

ENV_WORKSPACE_OWNER_FQCN = "NVFL_WORKSPACE_OWNER_FQCN"
ENV_WORKSPACE_TRANSFER_TOKEN = "NVFL_WORKSPACE_TRANSFER_TOKEN"

WORKSPACE_TRANSFER_CHANNEL = "workspace_transfer"
TOPIC_PREPARE_DOWNLOAD = "prepare_download"
TOPIC_PUBLISH_RESULTS = "publish_results"

DOWNLOAD_TIMEOUT = 600.0
PER_REQUEST_TIMEOUT = 300.0
BOOTSTRAP_CONNECT_TIMEOUT = 30.0
BOOTSTRAP_CONNECT_POLL_INTERVAL = 0.1

_BOOTSTRAP_CELL_PREFIX = "ws_transfer_"
_WORKSPACE_DOWNLOAD_EXCLUDES = frozenset({"local/study_data_pvc.yaml"})


@dataclass
class _JobTransferRecord:
    job_id: str
    workspace_root: str
    transfer_token: str
    download_tx_id: str = ""
    download_bundle_path: str = ""


def _write_dir_to_zip(zf: zipfile.ZipFile, src: str, root: str, excluded_paths: frozenset[str] = frozenset()) -> None:
    if not os.path.isdir(src):
        return
    for dirpath, _dirs, files in os.walk(src):
        for fname in files:
            abs_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")
            if rel_path in excluded_paths:
                continue
            zf.write(abs_path, rel_path)


def _zip_workspace_to_file(workspace_root: str, job_id: str, file_path: str) -> None:
    with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as zf:
        _write_dir_to_zip(zf, os.path.join(workspace_root, "local"), workspace_root, _WORKSPACE_DOWNLOAD_EXCLUDES)
        _write_dir_to_zip(zf, os.path.join(workspace_root, job_id), workspace_root)


def _zip_results_to_file(workspace_root: str, job_id: str, file_path: str) -> None:
    with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as zf:
        _write_dir_to_zip(zf, os.path.join(workspace_root, job_id), workspace_root)


def _validate_relative_zip_members(zf: zipfile.ZipFile) -> None:
    for info in zf.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe zip member: {info.filename}")


def _validate_job_zip_members(zf: zipfile.ZipFile, job_id: str) -> None:
    _validate_relative_zip_members(zf)
    for info in zf.infolist():
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"symlink not allowed in results archive: {info.filename}")
        parts = PurePosixPath(info.filename).parts
        if not parts or parts[0] != job_id:
            raise ValueError(f"zip member outside job workspace: {info.filename}")


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_workspace_transfer_fqcn(owner_fqcn: str, job_id: str) -> str:
    return FQCN.join([owner_fqcn, f"{_BOOTSTRAP_CELL_PREFIX}{job_id}"])


def _cleanup_files(paths) -> None:
    for path in paths:
        if not path:
            continue
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _cleanup_transfer_files(_tx_id: str, _status: str, _objects: list, temp_paths=None, **_kwargs) -> None:
    _cleanup_files(temp_paths or [])


def _cleanup_download(tx_id: str, bundle_path: str) -> None:
    if tx_id:
        DownloadService.delete_transaction(tx_id)
    if bundle_path:
        _cleanup_files([bundle_path])


def _make_error(message: str, rc: str = ReturnCode.INVALID_REQUEST) -> Message:
    logger.error(message)
    return make_reply(rc, error=message)


class WorkspaceTransferManager:
    """Manage per-job workspace transfer over an existing CellNet cell."""

    def __init__(
        self,
        cell: Cell,
        download_timeout: float = DOWNLOAD_TIMEOUT,
        per_request_timeout: float = PER_REQUEST_TIMEOUT,
    ):
        self.cell = cell
        self.owner_fqcn = cell.get_fqcn()
        self.download_timeout = download_timeout
        self.per_request_timeout = per_request_timeout
        self.jobs: dict[str, _JobTransferRecord] = {}
        self._lock = threading.Lock()

        self.cell.register_request_cb(
            channel=WORKSPACE_TRANSFER_CHANNEL,
            topic=TOPIC_PREPARE_DOWNLOAD,
            cb=self._handle_prepare_download,
        )
        self.cell.register_request_cb(
            channel=WORKSPACE_TRANSFER_CHANNEL,
            topic=TOPIC_PUBLISH_RESULTS,
            cb=self._handle_publish_results,
        )

    @classmethod
    def get_or_create(cls, cell: Cell) -> "WorkspaceTransferManager":
        """Return the per-cell manager, constructing one on first use."""
        lock = cell.__dict__.setdefault("_workspace_transfer_lock", threading.Lock())
        with lock:
            manager = cell.__dict__.get("_workspace_transfer_manager")
            if manager is None:
                manager = cls(cell)
                cell.__dict__["_workspace_transfer_manager"] = manager
            return manager

    def add_job(self, job_id: str, workspace_root: str) -> str:
        record = _JobTransferRecord(
            job_id=job_id,
            workspace_root=workspace_root,
            transfer_token=secrets.token_urlsafe(24),
        )
        with self._lock:
            old = self.jobs.get(job_id)
            self.jobs[job_id] = record
        if old is not None:
            _cleanup_download(old.download_tx_id, old.download_bundle_path)
        return record.transfer_token

    def remove_job(self, job_id: str) -> None:
        with self._lock:
            record = self.jobs.pop(job_id, None)
        if record is not None:
            _cleanup_download(record.download_tx_id, record.download_bundle_path)

    def _get_record(self, job_id: str) -> _JobTransferRecord | None:
        with self._lock:
            return self.jobs.get(job_id)

    def _authorize(self, request: Message, action: str) -> tuple[_JobTransferRecord | None, str, Message | None]:
        """Validate request payload and token. Returns (record, origin, error_reply)."""
        origin = request.get_header(MessageHeaderKey.ORIGIN) or ""
        payload = request.payload
        if not isinstance(payload, dict):
            return None, origin, _make_error(f"{action} payload must be dict")
        job_id = payload.get("job_id")
        if not job_id:
            return None, origin, _make_error(f"{action} missing job_id")
        transfer_token = payload.get("transfer_token")
        if not transfer_token:
            return None, origin, _make_error(f"{action} missing transfer_token")
        record = self._get_record(job_id)
        if not record:
            return None, origin, _make_error(f"unknown job_id for {action}: {job_id}", rc=ReturnCode.INVALID_TARGET)
        if not secrets.compare_digest(transfer_token, record.transfer_token):
            return None, origin, _make_error(f"{action} token mismatch for {job_id}", rc=ReturnCode.UNAUTHENTICATED)
        return record, origin, None

    def _handle_prepare_download(self, request: Message) -> Message:
        record, _origin, err = self._authorize(request, "prepare_download")
        if err is not None:
            return err
        job_id = record.job_id

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        tmp.close()
        downloader = None
        try:
            _zip_workspace_to_file(record.workspace_root, record.job_id, tmp.name)
            downloader = ObjectDownloader(
                cell=self.cell,
                timeout=self.download_timeout,
                num_receivers=1,
                transaction_done_cb=self._download_transaction_done,
                job_id=job_id,
            )
            ref_id = add_file(downloader, tmp.name)
            bundle_sha = _hash_file(tmp.name)
            bundle_size = os.path.getsize(tmp.name)
        except Exception as e:
            _cleanup_download(downloader.tx_id if downloader else "", tmp.name)
            return _make_error(f"failed to prepare workspace download for {job_id}: {e}")

        with self._lock:
            current = self.jobs.get(job_id)
            if not current:
                _cleanup_download(downloader.tx_id, tmp.name)
                return _make_error(f"job removed while preparing workspace download: {job_id}")
            old_tx_id, old_bundle_path = current.download_tx_id, current.download_bundle_path
            current.download_tx_id = downloader.tx_id
            current.download_bundle_path = tmp.name
        _cleanup_download(old_tx_id, old_bundle_path)

        return make_reply(
            ReturnCode.OK,
            body={
                "job_id": job_id,
                "ref_id": ref_id,
                "sha256": bundle_sha,
                "size": bundle_size,
            },
        )

    def _download_transaction_done(self, tx_id: str, _status: str, objects: list, job_id: str) -> None:
        _cleanup_files(objects)
        with self._lock:
            record = self.jobs.get(job_id)
            if not record or record.download_tx_id != tx_id:
                return
            record.download_tx_id = ""
            record.download_bundle_path = ""

    def _handle_publish_results(self, request: Message) -> Message:
        logger.info("[ws-transfer] publish_results handler entered on parent")
        record, origin, err = self._authorize(request, "publish_results")
        if err is not None:
            logger.info("[ws-transfer] publish_results rejected by authorize")
            return err
        job_id = record.job_id
        if not origin:
            return _make_error("publish_results missing request origin")
        payload = request.payload
        ref_id = payload.get("ref_id")
        if not ref_id:
            return _make_error("publish_results missing ref_id")
        logger.info("[ws-transfer] publish_results accepted for job=%s origin=%s ref=%s", job_id, origin, ref_id)

        expected_sha = payload.get("sha256")
        temp_dir = tempfile.mkdtemp(prefix="workspace-results-")
        try:
            err, file_path = download_file(
                from_fqcn=origin,
                ref_id=ref_id,
                per_request_timeout=self.per_request_timeout,
                cell=self.cell,
                location=temp_dir,
            )
            if err:
                logger.info("[ws-transfer] publish_results download_file failed for %s: %s", job_id, err)
                return _make_error(f"failed to download results for {job_id}: {err}", rc=ReturnCode.COMM_ERROR)
            if expected_sha and _hash_file(file_path) != expected_sha:
                return _make_error(f"results checksum mismatch for {job_id}")

            os.makedirs(record.workspace_root, exist_ok=True)
            with zipfile.ZipFile(file_path) as zf:
                _validate_job_zip_members(zf, record.job_id)
                zf.extractall(record.workspace_root)
            logger.info("[ws-transfer] publish_results extracted job=%s into %s", job_id, record.workspace_root)
        except ValueError as e:
            return _make_error(str(e))
        except zipfile.BadZipFile as e:
            return _make_error(f"invalid results bundle for {job_id}: {e}")
        except Exception as e:
            return _make_error(f"unexpected error processing results for {job_id}: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.remove_job(job_id)
        return make_reply(ReturnCode.OK, body={"job_id": job_id})


# Process-level singleton. download_workspace runs at startup and
# upload_results runs at shutdown inside the SAME process, so they share
# one bootstrap cell for the life of the job. Creating a second cell with
# the same FQCN crashes CellNet's registry ("there is already a cell
# named ..."), and nothing in CellNet unregisters the name reliably after
# cell.stop(), so keeping one alive is the simplest contract.
_bootstrap_cell: Cell | None = None
_bootstrap_net_agent: NetAgent | None = None
_bootstrap_lock = threading.Lock()


def _get_bootstrap_cell(args, owner_fqcn: str, secure_mode: bool) -> Cell:
    global _bootstrap_cell, _bootstrap_net_agent
    with _bootstrap_lock:
        if _bootstrap_cell is None:
            _bootstrap_cell, _bootstrap_net_agent = _create_bootstrap_cell(
                args=args, owner_fqcn=owner_fqcn, secure_mode=secure_mode
            )
        return _bootstrap_cell


def _close_bootstrap_cell() -> None:
    global _bootstrap_cell, _bootstrap_net_agent
    with _bootstrap_lock:
        if _bootstrap_net_agent is not None:
            try:
                _bootstrap_net_agent.close()
            except Exception:
                pass
            _bootstrap_net_agent = None
        if _bootstrap_cell is not None:
            try:
                _bootstrap_cell.stop()
            except Exception:
                pass
            _bootstrap_cell = None


def _get_root_url(args) -> str:
    root_url = getattr(args, "root_url", "")
    if root_url:
        return root_url
    scheme = getattr(args, "sp_scheme", "")
    target = getattr(args, "sp_target", "")
    if scheme and target:
        return f"{scheme}://{target}"
    raise RuntimeError("unable to determine root_url for workspace transfer bootstrap cell")


def _get_bootstrap_tls_pair(startup_dir: str, owner_fqcn: str) -> tuple[str, str, str, str]:
    prefer_server = FQCN.get_root(owner_fqcn) == FQCN.ROOT_SERVER
    if prefer_server:
        candidates = [
            ("server.crt", "server.key", DriverParams.SERVER_CERT.value, DriverParams.SERVER_KEY.value),
            ("client.crt", "client.key", DriverParams.CLIENT_CERT.value, DriverParams.CLIENT_KEY.value),
        ]
    else:
        candidates = [
            ("client.crt", "client.key", DriverParams.CLIENT_CERT.value, DriverParams.CLIENT_KEY.value),
            ("server.crt", "server.key", DriverParams.SERVER_CERT.value, DriverParams.SERVER_KEY.value),
        ]

    for cert_name, key_name, cert_key, key_key in candidates:
        cert_path = os.path.join(startup_dir, cert_name)
        key_path = os.path.join(startup_dir, key_name)
        if os.path.exists(cert_path) and os.path.exists(key_path):
            return cert_path, key_path, cert_key, key_key
    raise RuntimeError(f"workspace transfer requires cert/key files in startup dir: {startup_dir}")


def _create_bootstrap_cell(args, owner_fqcn: str, secure_mode: bool) -> tuple[Cell, NetAgent]:
    startup_dir = os.path.join(args.workspace, "startup")
    credentials = {}
    if secure_mode:
        root_ca = os.path.join(startup_dir, "rootCA.pem")
        if not os.path.exists(root_ca):
            raise RuntimeError(f"workspace transfer requires rootCA.pem in startup dir: {startup_dir}")
        cert_path, key_path, cert_key, key_key = _get_bootstrap_tls_pair(startup_dir, owner_fqcn)
        credentials = {
            DriverParams.CA_CERT.value: root_ca,
            cert_key: cert_path,
            key_key: key_path,
        }

    parent_resources = {}
    parent_conn_sec = getattr(args, "parent_conn_sec", "")
    if parent_conn_sec:
        parent_resources[DriverParams.CONNECTION_SECURITY.value] = parent_conn_sec

    fqcn = make_workspace_transfer_fqcn(owner_fqcn, args.job_id)
    cell = Cell(
        fqcn=fqcn,
        root_url=_get_root_url(args),
        secure=secure_mode,
        credentials=credentials,
        create_internal_listener=False,
        parent_url=args.parent_url,
        parent_resources=parent_resources or None,
    )
    # Install auth headers BEFORE cell.start(): the cell's initial
    # cellnet.channel registration handshake fires during start(), and the
    # parent's authenticator drops any unsigned message, preventing the cell
    # from ever registering with its parent.
    if FQCN.get_root(owner_fqcn) == FQCN.ROOT_SERVER:
        client_name = AUTH_CLIENT_NAME_FOR_SJ
        auth_token = args.job_id
    else:
        client_name = getattr(args, "client_name", "") or ""
        auth_token = getattr(args, "token", "") or ""
    set_add_auth_headers_filters(
        cell,
        client_name=client_name,
        auth_token=auth_token,
        token_signature=getattr(args, "token_signature", "") or "",
        ssid=getattr(args, "ssid", "") or None,
    )
    cell.start()
    net_agent = NetAgent(cell)
    return cell, net_agent


def _wait_for_bootstrap_ready(cell: Cell, owner_fqcn: str, timeout: float = BOOTSTRAP_CONNECT_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cell.is_backbone_ready() and cell.is_cell_connected(owner_fqcn):
            return
        time.sleep(BOOTSTRAP_CONNECT_POLL_INTERVAL)
    raise RuntimeError(
        f"workspace transfer bootstrap cell failed to connect to parent {owner_fqcn} within {timeout} seconds"
    )


def _request_workspace_bundle(cell: Cell, owner_fqcn: str, job_id: str, transfer_token: str) -> dict:
    _wait_for_bootstrap_ready(cell, owner_fqcn)
    request = new_cell_message({}, {"job_id": job_id, "transfer_token": transfer_token})
    reply = cell.send_request(
        channel=WORKSPACE_TRANSFER_CHANNEL,
        target=owner_fqcn,
        topic=TOPIC_PREPARE_DOWNLOAD,
        request=request,
        timeout=PER_REQUEST_TIMEOUT,
    )
    rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
    if rc != ReturnCode.OK:
        raise RuntimeError(f"workspace download preparation failed for {job_id}: {rc}")
    payload = reply.payload
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid workspace download reply payload for {job_id}: {type(payload)}")
    return payload


def download_workspace(args, secure_mode: bool) -> None:
    owner_fqcn = os.environ.get(ENV_WORKSPACE_OWNER_FQCN, "")
    if not owner_fqcn:
        return
    transfer_token = os.environ.get(ENV_WORKSPACE_TRANSFER_TOKEN, "")
    if not transfer_token:
        raise RuntimeError(f"workspace transfer requires env var {ENV_WORKSPACE_TRANSFER_TOKEN}")

    os.makedirs(args.workspace, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="workspace-download-")
    try:
        cell = _get_bootstrap_cell(args, owner_fqcn, secure_mode)
        payload = _request_workspace_bundle(cell, owner_fqcn, args.job_id, transfer_token)
        ref_id = payload.get("ref_id")
        expected_sha = payload.get("sha256")
        err, bundle_path = download_file(
            from_fqcn=owner_fqcn,
            ref_id=ref_id,
            per_request_timeout=PER_REQUEST_TIMEOUT,
            cell=cell,
            location=temp_dir,
        )
        if err:
            raise RuntimeError(f"failed to download workspace for {args.job_id}: {err}")
        if expected_sha and _hash_file(bundle_path) != expected_sha:
            raise RuntimeError(f"workspace bundle checksum mismatch for {args.job_id}")
        with zipfile.ZipFile(bundle_path) as zf:
            _validate_relative_zip_members(zf)
            zf.extractall(args.workspace)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def upload_results(args, secure_mode: bool) -> None:
    owner_fqcn = os.environ.get(ENV_WORKSPACE_OWNER_FQCN, "")
    if not owner_fqcn:
        return
    transfer_token = os.environ.get(ENV_WORKSPACE_TRANSFER_TOKEN, "")
    if not transfer_token:
        raise RuntimeError(f"workspace transfer requires env var {ENV_WORKSPACE_TRANSFER_TOKEN}")

    run_dir = os.path.join(args.workspace, args.job_id)
    if not os.path.isdir(run_dir):
        return

    temp_bundle = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    temp_bundle.close()
    downloader = None
    try:
        _zip_results_to_file(args.workspace, args.job_id, temp_bundle.name)
        bundle_sha = _hash_file(temp_bundle.name)
        bundle_size = os.path.getsize(temp_bundle.name)
        logger.info(
            "[ws-transfer] upload_results start job=%s bundle_size=%d target=%s",
            args.job_id,
            bundle_size,
            owner_fqcn,
        )
        cell = _get_bootstrap_cell(args, owner_fqcn, secure_mode)
        _wait_for_bootstrap_ready(cell, owner_fqcn)
        downloader = ObjectDownloader(
            cell=cell,
            timeout=DOWNLOAD_TIMEOUT,
            num_receivers=1,
            transaction_done_cb=_cleanup_transfer_files,
            temp_paths=[temp_bundle.name],
        )
        ref_id = add_file(downloader, temp_bundle.name)
        logger.info(
            "[ws-transfer] upload_results registered ref=%s tx=%s, sending publish_results",
            ref_id,
            getattr(downloader, "tx_id", "<unknown>"),
        )
        request = new_cell_message(
            {},
            {
                "job_id": args.job_id,
                "ref_id": ref_id,
                "transfer_token": transfer_token,
                "sha256": bundle_sha,
                "size": bundle_size,
            },
        )
        reply = cell.send_request(
            channel=WORKSPACE_TRANSFER_CHANNEL,
            target=owner_fqcn,
            topic=TOPIC_PUBLISH_RESULTS,
            request=request,
            timeout=DOWNLOAD_TIMEOUT,
        )
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        reply_origin = reply.get_header(MessageHeaderKey.ORIGIN)
        reply_err = reply.get_header(MessageHeaderKey.ERROR, "")
        logger.info(
            "[ws-transfer] upload_results reply rc=%s origin=%s err=%s payload=%r",
            rc,
            reply_origin,
            reply_err,
            getattr(reply, "payload", None),
        )
        if rc != ReturnCode.OK:
            raise RuntimeError(f"results upload failed for {args.job_id}: rc={rc} err={reply_err}")
        downloader.delete_transaction()
        downloader = None
        logger.info("[ws-transfer] upload_results SUCCESS job=%s", args.job_id)
    finally:
        if downloader is not None:
            downloader.delete_transaction()
        _cleanup_files([temp_bundle.name])
        # Cell is no longer needed after upload; final chance to free it.
        _close_bootstrap_cell()


def upload_results_safely(args, secure_mode: bool, log=None) -> None:
    try:
        upload_results(args, secure_mode)
    except Exception as e:
        (log or logger).warning(f"failed to upload job results for {args.job_id}: {secure_format_exception(e)}")

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
import shutil
import struct
import threading
import time
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.connection import BytesAlike, Connection
from nvflare.fuel.f3.drivers.base_driver import BaseDriver
from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo, Mode
from nvflare.fuel.f3.drivers.driver_params import DriverCap, DriverParams
from nvflare.fuel.f3.drivers.net_utils import MAX_FRAME_SIZE
from nvflare.fuel.f3.sfm.prefix import PREFIX_LEN
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.security.logging import secure_format_exception

FRAME_LEN_STRUCT = struct.Struct(">I")

# Resource/query parameter names
ROOT_DIR = "root_dir"
POLL_INTERVAL = "poll_interval"
MAX_POLL_INTERVAL = "max_poll_interval"
MAX_LOG_SIZE = "max_log_size"
LEASE_INTERVAL = "lease_interval"
LEASE_TIMEOUT = "lease_timeout"
FSYNC = "fsync"

# Tuning params propagated from listener resources to the connect URL so the active side uses the same values
_TUNING_PARAMS = [POLL_INTERVAL, MAX_POLL_INTERVAL, MAX_LOG_SIZE, LEASE_INTERVAL, LEASE_TIMEOUT, FSYNC]

# Directory layout
CONNS_DIR = "conns"
LISTENER_PREFIX = "lst_"
LISTENER_LEASE_FILE = "lease"
CLOSED_FILE = "closed"
TMP_PREFIX = "."
A2P = "a2p"  # active-to-passive log prefix
P2A = "p2a"  # passive-to-active log prefix

LISTENER_STALE_TIME = 600.0


def _touch(path: str):
    try:
        os.utime(path, None)
    except FileNotFoundError:
        try:
            open(path, "ab").close()
        except OSError:
            pass
    except OSError:
        pass


def _mtime(path: str) -> Optional[float]:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("true", "1", "yes", "y", "t")


class _ConnConfig:
    def __init__(self, params: dict):
        self.poll_interval = float(params.get(POLL_INTERVAL, 0.01))
        self.max_poll_interval = float(params.get(MAX_POLL_INTERVAL, 0.5))
        self.max_log_size = int(params.get(MAX_LOG_SIZE, 1024 * 1024 * 1024))
        self.lease_interval = float(params.get(LEASE_INTERVAL, 5.0))
        self.lease_timeout = float(params.get(LEASE_TIMEOUT, 30.0))
        self.fsync = _to_bool(params.get(FSYNC, False))


class _LogWriter:
    """Single-writer append log. Frames are appended verbatim; the SFM prefix provides framing.

    A frame never spans two log files: rotation only happens between appends. The final size of a
    rotated log is recorded in a ".sealed" file so the reader can tell "log complete" from
    "appends not yet visible" on a networked filesystem.
    """

    def __init__(self, conn_dir: str, prefix: str, cfg: _ConnConfig):
        self.conn_dir = conn_dir
        self.prefix = prefix
        self.cfg = cfg
        self.index = 0
        self.size = 0
        self.file = None
        self.closed = False
        self.lock = threading.Lock()

    def log_path(self, index: int) -> str:
        return os.path.join(self.conn_dir, f"{self.prefix}.{index}.log")

    def sealed_path(self, index: int) -> str:
        return os.path.join(self.conn_dir, f"{self.prefix}.{index}.sealed")

    def append(self, frame: BytesAlike):
        with self.lock:
            if self.closed:
                raise CommError(CommError.CLOSED, "Log writer is closed")
            if self.file is None:
                path = self.log_path(self.index)
                self.file = open(path, "ab")
                self.size = os.path.getsize(path)
            self.file.write(frame)
            self.file.flush()
            if self.cfg.fsync:
                os.fsync(self.file.fileno())
            self.size += len(frame)
            if self.size >= self.cfg.max_log_size:
                self._rotate()

    def _rotate(self):
        # fsync before sealing so the recorded size is never ahead of visible data
        os.fsync(self.file.fileno())
        self.file.close()
        sealed = self.sealed_path(self.index)
        tmp = sealed + ".tmp"
        with open(tmp, "w") as f:
            f.write(str(self.size))
        os.replace(tmp, sealed)
        self.index += 1
        self.file = open(self.log_path(self.index), "ab")
        self.size = 0

    def close(self):
        with self.lock:
            self.closed = True
            if self.file:
                try:
                    self.file.flush()
                    self.file.close()
                except OSError:
                    pass
                self.file = None


class _LogReader:
    """Tails the peer's append log, extracting whole frames and advancing across sealed logs."""

    def __init__(self, conn_dir: str, prefix: str, cfg: _ConnConfig):
        self.conn_dir = conn_dir
        self.prefix = prefix
        self.cfg = cfg
        self.index = 0
        self.offset = 0
        self.file = None
        self.buffer = bytearray()

    def log_path(self, index: int) -> str:
        return os.path.join(self.conn_dir, f"{self.prefix}.{index}.log")

    def sealed_path(self, index: int) -> str:
        return os.path.join(self.conn_dir, f"{self.prefix}.{index}.sealed")

    def read_frames(self) -> list:
        data = self._read_new_data()
        if data:
            self.buffer.extend(data)

        frames = []
        while True:
            frame = self._extract_frame()
            if frame is None:
                break
            frames.append(frame)

        self._advance_if_sealed()
        return frames

    def _read_new_data(self) -> Optional[bytes]:
        path = self.log_path(self.index)
        try:
            size = os.stat(path).st_size
        except OSError:
            # Not created yet (writer still opening the next log after rotation)
            return None
        if size <= self.offset:
            return None
        if self.file is None:
            self.file = open(path, "rb")
        self.file.seek(self.offset)
        data = self.file.read(size - self.offset)
        self.offset += len(data)
        return data

    def _extract_frame(self) -> Optional[bytes]:
        if len(self.buffer) < FRAME_LEN_STRUCT.size:
            return None
        length = FRAME_LEN_STRUCT.unpack_from(self.buffer, 0)[0]
        if length < PREFIX_LEN or length > MAX_FRAME_SIZE:
            raise CommError(CommError.BAD_DATA, f"Corrupt frame length {length} in {self.log_path(self.index)}")
        if len(self.buffer) < length:
            return None
        frame = bytes(self.buffer[:length])
        del self.buffer[:length]
        return frame

    def _advance_if_sealed(self):
        sealed = self.sealed_path(self.index)
        try:
            with open(sealed) as f:
                final_size = int(f.read())
        except (OSError, ValueError):
            return
        if self.offset < final_size:
            return
        if self.buffer:
            raise CommError(CommError.BAD_DATA, f"Partial frame at end of sealed log {self.log_path(self.index)}")
        if self.file:
            self.file.close()
            self.file = None
        for path in (self.log_path(self.index), sealed):
            try:
                os.unlink(path)
            except OSError:
                pass
        self.index += 1
        self.offset = 0

    def close(self):
        if self.file:
            try:
                self.file.close()
            except OSError:
                pass
            self.file = None


class FileConnection(Connection):
    """A connection emulated over a shared directory with one append log per direction."""

    def __init__(self, conn_dir: str, connector: ConnectorInfo, cfg: _ConnConfig):
        super().__init__(connector)
        self.conn_dir = conn_dir
        self.cfg = cfg
        self.closing = False
        self.logger = get_obj_logger(self)

        active = connector.mode == Mode.ACTIVE
        self.side = "a" if active else "p"
        peer_side = "p" if active else "a"
        self.my_lease = os.path.join(conn_dir, f"{self.side}.lease")
        self.peer_lease = os.path.join(conn_dir, f"{peer_side}.lease")
        self.closed_file = os.path.join(conn_dir, CLOSED_FILE)
        self.writer = _LogWriter(conn_dir, A2P if active else P2A, cfg)
        self.reader = _LogReader(conn_dir, P2A if active else A2P, cfg)

        _touch(self.my_lease)
        self.conn_props = {
            DriverParams.LOCAL_ADDR.value: f"{conn_dir}#{self.side}",
            DriverParams.PEER_ADDR.value: f"{conn_dir}#{peer_side}",
        }

    def get_conn_properties(self) -> dict:
        return self.conn_props

    def close(self):
        if self.closing:
            return
        self.closing = True
        _touch(self.closed_file)
        self.writer.close()

    def send_frame(self, frame: BytesAlike):
        if self.closing:
            raise CommError(CommError.CLOSED, f"Connection {self.name} is closed")
        try:
            self.writer.append(frame)
        except CommError:
            raise
        except Exception as ex:
            raise CommError(CommError.ERROR, f"Error sending frame: {secure_format_exception(ex)}")

    def read_loop(self, stopped: threading.Event):
        """Poll the incoming log and deliver frames until the connection ends.

        Liveness is tracked by observed changes of the peer's lease mtime (not absolute file
        times), so it is immune to clock skew between nodes.
        """
        interval = self.cfg.poll_interval
        now = time.monotonic()
        last_housekeeping = now
        peer_seen = now
        peer_mtime = None

        while not self.closing and not stopped.is_set():
            now = time.monotonic()
            if now - last_housekeeping >= self.cfg.lease_interval:
                last_housekeeping = now
                _touch(self.my_lease)
                if os.path.exists(self.closed_file):
                    self.logger.debug(f"{self}: closed by peer")
                    break
                mtime = _mtime(self.peer_lease)
                if mtime is not None and mtime != peer_mtime:
                    peer_mtime = mtime
                    peer_seen = now
                elif now - peer_seen > self.cfg.lease_timeout:
                    self.logger.info(f"{self}: peer lease is stale, closing")
                    break

            frames = self.reader.read_frames()
            if frames:
                peer_seen = now
                interval = self.cfg.poll_interval
                for frame in frames:
                    self.process_frame(frame)
            else:
                interval = min(interval * 1.5, self.cfg.max_poll_interval)
                stopped.wait(interval)

        self.reader.close()


class FileDriver(BaseDriver):
    """Transport driver that exchanges SFM frames through a shared filesystem (e.g. Lustre).

    Intended for deployments where two cells share a filesystem but have no network path between
    them (e.g. an HPC compute node and a gateway node). The URL form is file://0/absolute/dir
    ("0" is the empty-host placeholder). The passive side owns the directory; each active peer
    creates a connection subdirectory with one append log per direction. There is no transport
    security: directory permissions are the trust boundary.
    """

    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.dir_lock = threading.Lock()
        self.handled_dirs = set()
        self.done_dirs = set()
        self.logger = get_obj_logger(self)

    @staticmethod
    def supported_transports() -> List[str]:
        return ["file"]

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {DriverCap.SEND_HEARTBEAT.value: True, DriverCap.SUPPORT_SSL.value: False}

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):
        root_dir = resources.get(ROOT_DIR)
        if not root_dir:
            raise CommError(CommError.BAD_CONFIG, f"'{ROOT_DIR}' resource is required for scheme {scheme}")
        root_dir = os.path.abspath(root_dir)
        os.makedirs(root_dir, exist_ok=True)
        _remove_stale_listener_dirs(root_dir)

        listen_dir = os.path.join(root_dir, LISTENER_PREFIX + uuid.uuid4().hex[:8])
        os.makedirs(os.path.join(listen_dir, CONNS_DIR))

        url = f"{scheme}://0{listen_dir}"
        tuning = {k: str(resources[k]) for k in _TUNING_PARAMS if k in resources}
        if tuning:
            url += "?" + urlencode(tuning)
        return url, url

    def listen(self, connector: ConnectorInfo):
        self.connector = connector
        cfg = _ConnConfig(connector.params)
        listen_dir = self._get_dir_from_params(connector.params)
        conns_dir = os.path.join(listen_dir, CONNS_DIR)
        os.makedirs(conns_dir, exist_ok=True)
        lease = os.path.join(listen_dir, LISTENER_LEASE_FILE)

        interval = cfg.poll_interval
        last_housekeeping = 0.0

        while not self._stopping(connector):
            now = time.monotonic()
            if now - last_housekeeping >= cfg.lease_interval:
                last_housekeeping = now
                _touch(lease)
                self._gc_conn_dirs(conns_dir, cfg)

            new_conns = self._scan_for_connections(conns_dir, connector, cfg)
            if new_conns:
                interval = cfg.poll_interval
            else:
                interval = min(interval * 1.5, cfg.max_poll_interval)
                connector.stopped.wait(interval)

    def connect(self, connector: ConnectorInfo):
        self.connector = connector
        cfg = _ConnConfig(connector.params)
        listen_dir = self._get_dir_from_params(connector.params)
        conns_dir = os.path.join(listen_dir, CONNS_DIR)
        if not os.path.isdir(conns_dir):
            raise CommError(CommError.NOT_READY, f"Listener dir is not ready: {conns_dir}")

        name = uuid.uuid4().hex[:12]
        tmp_dir = os.path.join(conns_dir, TMP_PREFIX + name)
        os.makedirs(tmp_dir)
        for log_name in (f"{A2P}.0.log", f"{P2A}.0.log"):
            open(os.path.join(tmp_dir, log_name), "ab").close()
        conn_dir = os.path.join(conns_dir, name)
        os.rename(tmp_dir, conn_dir)

        conn = FileConnection(conn_dir, connector, cfg)
        self.add_connection(conn)
        try:
            conn.read_loop(self.stop_event)
        finally:
            conn.close()
            self.close_connection(conn)

    def shutdown(self):
        self.stop_event.set()
        self.close_all()

    # Internal methods

    def _stopping(self, connector: ConnectorInfo) -> bool:
        return self.stop_event.is_set() or connector.stopped.is_set()

    @staticmethod
    def _get_dir_from_params(params: dict) -> str:
        path = params.get(DriverParams.PATH.value)
        if not path or path == "/":
            url = params.get(DriverParams.URL.value)
            raise CommError(CommError.BAD_CONFIG, f"Missing directory path in URL {url}, expected file://0/abs/dir")
        return path

    def _scan_for_connections(self, conns_dir: str, connector: ConnectorInfo, cfg: _ConnConfig) -> int:
        try:
            entries = sorted(os.listdir(conns_dir))
        except OSError as ex:
            raise CommError(CommError.ERROR, f"Cannot list {conns_dir}: {secure_format_exception(ex)}")

        new_conns = 0
        for entry in entries:
            if entry.startswith(TMP_PREFIX):
                continue
            with self.dir_lock:
                if entry in self.handled_dirs:
                    continue
                self.handled_dirs.add(entry)
            conn_dir = os.path.join(conns_dir, entry)
            if not os.path.isdir(conn_dir):
                continue
            conn = FileConnection(conn_dir, connector, cfg)
            self.add_connection(conn)
            t = threading.Thread(target=self._conn_loop, args=(conn, entry), name=f"file_conn_{entry}", daemon=True)
            t.start()
            new_conns += 1
        return new_conns

    def _conn_loop(self, conn: FileConnection, entry: str):
        try:
            conn.read_loop(self.stop_event)
        except Exception as ex:
            self.logger.error(f"Connection {conn} closed due to error: {secure_format_exception(ex)}")
        finally:
            conn.close()
            self.close_connection(conn)
            with self.dir_lock:
                self.done_dirs.add(entry)

    def _gc_conn_dirs(self, conns_dir: str, cfg: _ConnConfig):
        """Remove connection dirs whose connection has ended, once the closed marker is old
        enough for the peer to have noticed it."""
        with self.dir_lock:
            done = list(self.done_dirs)

        for entry in done:
            conn_dir = os.path.join(conns_dir, entry)
            if os.path.isdir(conn_dir):
                closed_mtime = _mtime(os.path.join(conn_dir, CLOSED_FILE))
                if closed_mtime is None:
                    _touch(os.path.join(conn_dir, CLOSED_FILE))
                    continue
                if time.time() - closed_mtime <= 2 * cfg.lease_timeout:
                    continue
                shutil.rmtree(conn_dir, ignore_errors=True)
            with self.dir_lock:
                self.done_dirs.discard(entry)
                self.handled_dirs.discard(entry)


def _remove_stale_listener_dirs(root_dir: str):
    """GC listener dirs from previous runs whose lease has not been touched for a long time"""
    try:
        entries = os.listdir(root_dir)
    except OSError:
        return

    for entry in entries:
        if not entry.startswith(LISTENER_PREFIX):
            continue
        listen_dir = os.path.join(root_dir, entry)
        mtime = _mtime(os.path.join(listen_dir, LISTENER_LEASE_FILE))
        if mtime is None:
            mtime = _mtime(listen_dir)
        if mtime is None or time.time() - mtime > LISTENER_STALE_TIME:
            shutil.rmtree(listen_dir, ignore_errors=True)

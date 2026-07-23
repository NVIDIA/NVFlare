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
import logging
import os
import shutil
import struct
import threading
import time
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlsplit

from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.connection import BytesAlike, Connection
from nvflare.fuel.f3.drivers.base_driver import BaseDriver
from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo, Mode
from nvflare.fuel.f3.drivers.driver_params import DriverCap, DriverParams
from nvflare.fuel.f3.drivers.net_utils import MAX_FRAME_SIZE
from nvflare.fuel.f3.sfm.prefix import PREFIX_LEN
from nvflare.fuel.utils.argument_utils import str2bool
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.security.logging import secure_format_exception

log = logging.getLogger(__name__)

FRAME_LEN_STRUCT = struct.Struct(">I")

# Resource/query parameter names
ROOT_DIR = "root_dir"
POLL_INTERVAL = "poll_interval"
MAX_POLL_INTERVAL = "max_poll_interval"
MAX_LOG_SIZE = "max_log_size"
LEASE_INTERVAL = "lease_interval"
LEASE_TIMEOUT = "lease_timeout"
FSYNC = "fsync"
DIR_MODE = "dir_mode"
FILE_MODE = "file_mode"

# Tuning params propagated from listener resources to the connect URL so the active side uses the same values
_TUNING_PARAMS = [
    POLL_INTERVAL,
    MAX_POLL_INTERVAL,
    MAX_LOG_SIZE,
    LEASE_INTERVAL,
    LEASE_TIMEOUT,
    FSYNC,
    FILE_MODE,
    DIR_MODE,
]

# Directory layout
CONNS_DIR = "conns"
LISTENER_PREFIX = "lst_"
LISTENER_LEASE_FILE = "lease"
OWNER_MARKER = ".nvf_file_transport"
CLOSED_FILE = "closed"
TMP_PREFIX = "."
A2P = "a2p"  # active-to-passive log prefix
P2A = "p2a"  # passive-to-active log prefix

LISTENER_STALE_TIME = 3600.0
TMP_DIR_STALE_TIME = 3600.0
MIN_POLL_INTERVAL = 0.001
READ_CHUNK = 8 * 1024 * 1024
# After the peer's closed marker, keep draining until this many consecutive quiet polls; covers
# chunked reads of a large final frame and short cross-file visibility lag on networked FS
DRAIN_GRACE_POLLS = 20
DRAIN_POLL_INTERVAL = 0.05


def parse_file_url(url: str) -> str:
    """Parse and validate a file transport URL of the form file://0/absolute/dir.

    The authority must be the empty-host placeholder "0" and the path must be an absolute
    directory. This is the single source of truth for file URL semantics; launchers and deploy
    tools that need the directory (e.g. to bind-mount it into a container) should use this
    instead of parsing the URL themselves.

    Returns:
        The absolute directory path (query parameters excluded).

    Raises:
        CommError: If the URL is not a valid file transport URL.
    """
    parsed = urlsplit(url)
    if parsed.scheme != "file":
        raise CommError(CommError.BAD_CONFIG, f"Not a file transport URL: {url}")
    if parsed.netloc != "0":
        raise CommError(
            CommError.BAD_CONFIG, f"Invalid file URL {url}: authority must be the placeholder '0' (file://0/abs/dir)"
        )
    path = parsed.path.rstrip("/")
    if not path or not os.path.isabs(path):
        raise CommError(CommError.BAD_CONFIG, f"Invalid file URL {url}: an absolute directory path is required")
    return path


def _validate_dir_path(path: str):
    for ch in ("?", "#", "\n", "\r"):
        if ch in path:
            raise CommError(
                CommError.BAD_CONFIG, f"Unsupported character {ch!r} in file transport directory path: {path}"
            )


def _touch(path: str, mode: Optional[int] = None):
    try:
        os.utime(path, None)
    except FileNotFoundError:
        try:
            with open(path, "ab"):
                pass
            if mode is not None:
                os.chmod(path, mode)
        except OSError as ex:
            log.debug(f"Cannot create {path}: {secure_format_exception(ex)}")
    except OSError as ex:
        log.debug(f"Cannot touch {path}: {secure_format_exception(ex)}")


def _mtime(path: str) -> Optional[float]:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


def _make_new_dir(path: str, mode: int):
    os.makedirs(path)
    os.chmod(path, mode)


def _create_file(path: str, mode: int):
    with open(path, "ab"):
        pass
    os.chmod(path, mode)


def _log_path(conn_dir: str, prefix: str, index: int) -> str:
    return os.path.join(conn_dir, f"{prefix}.{index}.log")


def _sealed_path(conn_dir: str, prefix: str, index: int) -> str:
    return os.path.join(conn_dir, f"{prefix}.{index}.sealed")


def _parse_mode(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    return int(str(value), 8)


class _ConnConfig:
    """Effective tuning values for one connection.

    Precedence: hardcoded defaults < connector params (listener resources / URL query) < the
    local comm_config "file" section, so a cell can locally override polling/lease/fsync for its
    own mount. max_log_size is exempt from local override because reader-side rotation detection
    requires the same value on both sides of a connection.
    """

    def __init__(self, params: dict):
        merged = dict(params)
        config = CommConfigurator().get_config()
        local_overrides = config.get("file") if config else None
        if local_overrides:
            for key, value in local_overrides.items():
                if key != MAX_LOG_SIZE:
                    merged[key] = value

        self.poll_interval = max(float(merged.get(POLL_INTERVAL, 0.01)), MIN_POLL_INTERVAL)
        self.max_poll_interval = max(float(merged.get(MAX_POLL_INTERVAL, 2.0)), self.poll_interval)
        self.max_log_size = int(float(params.get(MAX_LOG_SIZE, 1024 * 1024 * 1024)))
        self.lease_interval = float(merged.get(LEASE_INTERVAL, 15.0))
        self.lease_timeout = float(merged.get(LEASE_TIMEOUT, 60.0))
        self.fsync = bool(str2bool(merged.get(FSYNC, False)))
        self.dir_mode = _parse_mode(merged.get(DIR_MODE), 0o770)
        self.file_mode = _parse_mode(merged.get(FILE_MODE), 0o660)


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

    def append(self, frame: BytesAlike):
        with self.lock:
            if self.closed:
                raise CommError(CommError.CLOSED, "Log writer is closed")
            if self.file is None:
                path = _log_path(self.conn_dir, self.prefix, self.index)
                existed = os.path.exists(path)
                self.file = open(path, "ab")
                if not existed:
                    os.chmod(path, self.cfg.file_mode)
                self.size = os.path.getsize(path)
            self.file.write(frame)
            self.file.flush()
            if self.cfg.fsync:
                os.fsync(self.file.fileno())
            self.size += len(frame)
            if self.size >= self.cfg.max_log_size:
                self._rotate()

    def _rotate(self):
        # fsync before sealing so the recorded size is never ahead of visible data. self.file is
        # cleared before any fallible step so a failed rotation never wedges the writer on a
        # closed handle; the caller closes the connection on error and a reconnect starts clean.
        os.fsync(self.file.fileno())
        self.file.close()
        self.file = None
        sealed = _sealed_path(self.conn_dir, self.prefix, self.index)
        tmp = sealed + ".tmp"
        with open(tmp, "w") as f:
            f.write(str(self.size))
        os.chmod(tmp, self.cfg.file_mode)
        os.replace(tmp, sealed)
        next_path = _log_path(self.conn_dir, self.prefix, self.index + 1)
        self.index += 1
        self.file = open(next_path, "ab")
        os.chmod(next_path, self.cfg.file_mode)
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
        self.progressed = False

    def read_frames(self) -> list:
        self.progressed = False
        data = self._read_new_data()
        if data:
            self.buffer.extend(data)
            self.progressed = True

        frames = []
        pos = 0
        buf = self.buffer
        total = len(buf)
        while total - pos >= FRAME_LEN_STRUCT.size:
            length = FRAME_LEN_STRUCT.unpack_from(buf, pos)[0]
            if length < PREFIX_LEN or length > MAX_FRAME_SIZE:
                raise CommError(
                    CommError.BAD_DATA,
                    f"Corrupt frame length {length} in {_log_path(self.conn_dir, self.prefix, self.index)}",
                )
            if total - pos < length:
                break
            frames.append(bytes(memoryview(buf)[pos : pos + length]))
            pos += length
        if pos:
            del buf[:pos]

        # A seal is only ever written at or past max_log_size, so skip the probe until then.
        # This requires max_log_size to be identical on both sides of a connection.
        if self.offset >= self.cfg.max_log_size:
            self._advance_if_sealed()
        return frames

    def _read_new_data(self) -> Optional[bytes]:
        path = _log_path(self.conn_dir, self.prefix, self.index)
        try:
            size = os.stat(path).st_size
            if size <= self.offset:
                return None
            if self.file is None:
                self.file = open(path, "rb")
            self.file.seek(self.offset)
            data = self.file.read(min(size - self.offset, READ_CHUNK))
        except OSError:
            # Not created yet (writer still opening the next log after rotation), or transient
            # FS error. Drop any open handle so a stale descriptor (e.g. ESTALE after NFS server
            # recovery) is replaced by a fresh open on the next poll instead of wedging forever.
            self.close()
            return None
        if data:
            self.offset += len(data)
        return data

    def _advance_if_sealed(self):
        sealed = _sealed_path(self.conn_dir, self.prefix, self.index)
        try:
            with open(sealed) as f:
                final_size = int(f.read())
        except (OSError, ValueError):
            return
        if self.offset < final_size:
            return
        if self.buffer:
            raise CommError(
                CommError.BAD_DATA,
                f"Partial frame at end of sealed log {_log_path(self.conn_dir, self.prefix, self.index)}",
            )
        if self.file:
            self.file.close()
            self.file = None
        for path in (_log_path(self.conn_dir, self.prefix, self.index), sealed):
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
        self.received_anything = False
        self.logger = get_obj_logger(self)

        active = connector.mode == Mode.ACTIVE
        side = "a" if active else "p"
        peer_side = "p" if active else "a"
        self.my_lease = os.path.join(conn_dir, f"{side}.lease")
        self.peer_lease = os.path.join(conn_dir, f"{peer_side}.lease")
        self.closed_file = os.path.join(conn_dir, CLOSED_FILE)
        self.writer = _LogWriter(conn_dir, A2P if active else P2A, cfg)
        self.reader = _LogReader(conn_dir, P2A if active else A2P, cfg)

        _touch(self.my_lease, cfg.file_mode)
        self.conn_props = {
            DriverParams.LOCAL_ADDR.value: f"{conn_dir}#{side}",
            DriverParams.PEER_ADDR.value: f"{conn_dir}#{peer_side}",
        }

    def get_conn_properties(self) -> dict:
        return self.conn_props

    def close(self):
        if self.closing:
            return
        self.closing = True
        # Close the writer before creating the closed marker so the marker always follows all
        # data: any in-flight append completes under the writer lock, later appends are rejected,
        # and a peer that sees the marker can trust a final drain to observe every frame.
        self.writer.close()
        _touch(self.closed_file, self.cfg.file_mode)

    def send_frame(self, frame: BytesAlike):
        if self.closing:
            raise CommError(CommError.CLOSED, f"Connection {self.name} is closed")
        try:
            self.writer.append(frame)
        except Exception as ex:
            # A failed append may have left a torn frame in the log; the framing can never
            # recover, so close the connection to force a clean reconnect.
            self.close()
            if isinstance(ex, CommError):
                raise
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
        peer_active = False

        try:
            while not self.closing and not stopped.is_set() and not self.connector.stopped.is_set():
                now = time.monotonic()
                if now - last_housekeeping >= self.cfg.lease_interval:
                    last_housekeeping = now
                    _touch(self.my_lease, self.cfg.file_mode)
                    if peer_active:
                        # Frames arrived since the last housekeeping, so the peer is alive; skip
                        # the peer-lease and closed-marker stats. A close mid-traffic is caught
                        # once the stream goes quiet (the marker follows all data anyway).
                        peer_active = False
                        peer_seen = now
                        peer_mtime = None
                    elif os.path.exists(self.closed_file):
                        self.logger.debug(f"{self}: closed by peer")
                        self._drain(stopped)
                        break
                    else:
                        mtime = _mtime(self.peer_lease)
                        if mtime is not None and mtime != peer_mtime:
                            peer_mtime = mtime
                            peer_seen = now
                        elif now - peer_seen > self.cfg.lease_timeout:
                            self.logger.info(f"{self}: peer lease is stale, closing")
                            break

                frames = self.reader.read_frames()
                if frames:
                    self.received_anything = True
                    peer_seen = now
                    peer_active = True
                    for frame in frames:
                        self.process_frame(frame)
                if frames or self.reader.progressed:
                    interval = self.cfg.poll_interval
                else:
                    interval = min(interval * 1.5, self.cfg.max_poll_interval)
                if not frames:
                    stopped.wait(interval)
        finally:
            self.reader.close()

    def _drain(self, stopped: threading.Event):
        """Deliver frames still in the log after the peer's closed marker (which follows all
        data). Reads are chunked, so a large final frame spans several read calls that return no
        complete frame yet — keep going while bytes are being consumed, and only give up after
        DRAIN_GRACE_POLLS consecutive quiet polls (grace for delayed visibility on networked FS).
        """
        quiet_polls = 0
        while quiet_polls < DRAIN_GRACE_POLLS:
            frames = self.reader.read_frames()
            if frames:
                self.received_anything = True
                for frame in frames:
                    self.process_frame(frame)
            if frames or self.reader.progressed:
                quiet_polls = 0
            else:
                quiet_polls += 1
                stopped.wait(min(self.cfg.poll_interval, DRAIN_POLL_INTERVAL))


class FileDriver(BaseDriver):
    """Transport driver that exchanges SFM frames through a shared filesystem (e.g. Lustre).

    Intended for deployments where two cells share a filesystem but have no network path between
    them (e.g. an HPC compute node and a gateway node). The URL form is file://0/absolute/dir
    ("0" is the empty-host placeholder). The passive side owns the directory; each active peer
    creates a connection subdirectory with one append log per direction. There is no transport
    security: directory permissions are the trust boundary. Directories are created 0o770 and
    files 0o660 regardless of umask; override with the dir_mode/file_mode resources (octal
    strings) if the two sides run as different users needing other group semantics.

    Typical use is job-cell to client-parent communication, configured in the client's
    comm_config.json. connect_generation 1 routes job-cell traffic through the client parent
    instead of dialing the server directly:

        {
            "backbone": {"connect_generation": 1},
            "internal": {
                "scheme": "file",
                "resources": {
                    "root_dir": "/absolute/shared/path/cellnet",
                    "connection_security": "clear",
                    "poll_interval": 0.05,
                    "max_poll_interval": 0.5,
                    "lease_interval": 5,
                    "lease_timeout": 30,
                    "fsync": false
                }
            }
        }

    The tuning resources are optional (defaults in _ConnConfig) and are propagated to the child
    cell through the query string of the generated connect URL; a cell can locally override them
    (except max_log_size, which must match on both sides) via a "file" section in its own
    comm_config.json. The example above pins lower-latency polling than the defaults.

    Filesystem metadata load: an idle connection costs one log stat per poll tick per direction
    (2/max_poll_interval per second) plus roughly 3 lease/marker ops per lease_interval per side,
    about 1.4 client-side metadata syscalls/sec per idle connection at the defaults
    (max_poll_interval=2, lease_interval=15) — these are client syscall counts; caching may or
    may not absorb them server-side. A listener adds ~0.5 listdir/sec. Costs shrink
    proportionally as max_poll_interval grows, at the price of first-message-after-idle latency.
    During traffic the peer-lease and closed-marker checks are skipped, data is read in 8MB
    chunks without sleeping, and stat frequency tracks frame arrival, so throughput is unaffected
    by poll settings. Note for NFS: attribute caching (actimeo) can delay lease-change and size
    visibility; mount with a low actimeo or raise lease_timeout well above the attribute-cache
    window. fsync=true is recommended on filesystems without close-to-open or POSIX coherence.
    """

    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.dir_lock = threading.Lock()
        self.handled_dirs = set()
        self.done_dirs: Dict[str, float] = {}
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
        _validate_dir_path(root_dir)
        cfg = _ConnConfig(resources)
        if not os.path.isdir(root_dir):
            _make_new_dir(root_dir, cfg.dir_mode)
        _remove_stale_listener_dirs(root_dir)

        listen_dir = os.path.join(root_dir, LISTENER_PREFIX + uuid.uuid4().hex[:8])
        _make_new_dir(listen_dir, cfg.dir_mode)
        _make_new_dir(os.path.join(listen_dir, CONNS_DIR), cfg.dir_mode)
        _create_file(os.path.join(listen_dir, OWNER_MARKER), cfg.file_mode)

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
        if not os.path.isdir(conns_dir):
            os.makedirs(conns_dir, exist_ok=True)
            os.chmod(conns_dir, cfg.dir_mode)
        lease = os.path.join(listen_dir, LISTENER_LEASE_FILE)

        interval = cfg.poll_interval
        last_housekeeping = 0.0

        while not self._stopping(connector):
            try:
                entries = sorted(os.listdir(conns_dir))
            except OSError as ex:
                if not os.path.isdir(conns_dir):
                    raise CommError(CommError.ERROR, f"Listener dir is gone: {conns_dir}")
                # Transient shared-FS error (e.g. ESTALE, brief server blip): keep the accept
                # loop alive and retry after a poll interval
                self.logger.debug(f"Cannot list {conns_dir}: {secure_format_exception(ex)}")
                connector.stopped.wait(cfg.max_poll_interval)
                continue

            now = time.monotonic()
            if now - last_housekeeping >= cfg.lease_interval:
                last_housekeeping = now
                _touch(lease, cfg.file_mode)
                self._gc_conn_dirs(conns_dir, cfg, entries)

            new_conns = self._adopt_new_dirs(conns_dir, entries, connector, cfg)
            if new_conns:
                interval = cfg.poll_interval
            else:
                interval = min(interval * 1.5, cfg.max_poll_interval)
                connector.stopped.wait(interval)

        self._cleanup_listen_dir(listen_dir)

    def connect(self, connector: ConnectorInfo):
        self.connector = connector
        cfg = _ConnConfig(connector.params)
        listen_dir = self._get_dir_from_params(connector.params)
        conns_dir = os.path.join(listen_dir, CONNS_DIR)
        if not os.path.isdir(conns_dir):
            raise CommError(CommError.NOT_READY, f"Listener dir is not ready: {conns_dir}")

        name = uuid.uuid4().hex[:12]
        tmp_dir = os.path.join(conns_dir, TMP_PREFIX + name)
        _make_new_dir(tmp_dir, cfg.dir_mode)
        for log_name in (f"{A2P}.0.log", f"{P2A}.0.log"):
            _create_file(os.path.join(tmp_dir, log_name), cfg.file_mode)
        conn_dir = os.path.join(conns_dir, name)
        os.rename(tmp_dir, conn_dir)

        conn = FileConnection(conn_dir, connector, cfg)
        self.add_connection(conn)
        try:
            conn.read_loop(self.stop_event)
        finally:
            conn.close()
            self.close_connection(conn)
            if not conn.received_anything:
                # Nothing ever arrived (e.g. the listener is dead); remove our own dir so
                # reconnect attempts don't accumulate directories on the shared filesystem
                shutil.rmtree(conn_dir, ignore_errors=True)

    def shutdown(self):
        self.stop_event.set()
        self.close_all()

    # Internal methods

    def _stopping(self, connector: ConnectorInfo) -> bool:
        return self.stop_event.is_set() or connector.stopped.is_set()

    @staticmethod
    def _get_dir_from_params(params: dict) -> str:
        url = params.get(DriverParams.URL.value)
        if url:
            return parse_file_url(url)
        path = params.get(DriverParams.PATH.value)
        if path:
            path = path.rstrip("/")
        if not path or not os.path.isabs(path):
            raise CommError(CommError.BAD_CONFIG, "Missing directory path, expected URL of form file://0/abs/dir")
        return path

    def _adopt_new_dirs(self, conns_dir: str, entries: list, connector: ConnectorInfo, cfg: _ConnConfig) -> int:
        new_conns = 0
        for entry in entries:
            if entry.startswith(TMP_PREFIX):
                continue
            with self.dir_lock:
                if entry in self.handled_dirs:
                    continue
            conn_dir = os.path.join(conns_dir, entry)
            if not os.path.isdir(conn_dir):
                continue
            if os.path.exists(os.path.join(conn_dir, CLOSED_FILE)) or not os.path.isfile(
                os.path.join(conn_dir, f"{A2P}.0.log")
            ):
                # Leftover from a previous listener incarnation: already closed, or its logs were
                # partially consumed. Adopting it would replay or wedge; mark for GC instead.
                _touch(os.path.join(conn_dir, CLOSED_FILE), cfg.file_mode)
                with self.dir_lock:
                    self.handled_dirs.add(entry)
                    self.done_dirs[entry] = time.monotonic()
                continue
            with self.dir_lock:
                self.handled_dirs.add(entry)
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
                self.done_dirs[entry] = time.monotonic()

    def _gc_conn_dirs(self, conns_dir: str, cfg: _ConnConfig, entries: list):
        """Remove ended connection dirs after a locally-timed grace period (no cross-node clock
        comparison), plus orphaned tmp dirs from crashed dialers."""
        with self.dir_lock:
            done = list(self.done_dirs.items())

        now = time.monotonic()
        for entry, done_time in done:
            if now - done_time <= 2 * cfg.lease_timeout:
                continue
            conn_dir = os.path.join(conns_dir, entry)
            shutil.rmtree(conn_dir, ignore_errors=True)
            if os.path.isdir(conn_dir):
                # Removal failed (e.g. peer still holds files open on NFS); keep the entry so the
                # dir is retried later and never re-adopted as a new connection
                continue
            with self.dir_lock:
                self.done_dirs.pop(entry, None)
                self.handled_dirs.discard(entry)

        for entry in entries:
            if not entry.startswith(TMP_PREFIX):
                continue
            tmp_dir = os.path.join(conns_dir, entry)
            mtime = _mtime(tmp_dir)
            if mtime is not None and time.time() - mtime > TMP_DIR_STALE_TIME:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _cleanup_listen_dir(self, listen_dir: str):
        """On clean shutdown, remove the listener dir if this driver's get_urls() minted it"""
        if os.path.basename(listen_dir).startswith(LISTENER_PREFIX) and os.path.isfile(
            os.path.join(listen_dir, OWNER_MARKER)
        ):
            shutil.rmtree(listen_dir, ignore_errors=True)


def _remove_stale_listener_dirs(root_dir: str):
    """GC listener dirs from previous runs whose lease has not been touched for a long time.

    Only directories carrying the driver's ownership marker are ever removed, so a mistakenly
    broad root_dir cannot cause loss of unrelated data. Removal runs in a background thread to
    keep cell startup off the critical path of large tree deletions.
    """
    try:
        entries = os.listdir(root_dir)
    except OSError:
        return

    for entry in entries:
        if not entry.startswith(LISTENER_PREFIX):
            continue
        listen_dir = os.path.join(root_dir, entry)
        if not os.path.isfile(os.path.join(listen_dir, OWNER_MARKER)):
            continue
        mtime = _mtime(os.path.join(listen_dir, LISTENER_LEASE_FILE))
        if mtime is None:
            mtime = _mtime(listen_dir)
        if mtime is None or time.time() - mtime > LISTENER_STALE_TIME:
            threading.Thread(
                target=shutil.rmtree, args=(listen_dir,), kwargs={"ignore_errors": True}, daemon=True
            ).start()

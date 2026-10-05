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


"""Ordinary trusted participant streaming CC diagnostics; no token or error text export."""

import datetime
import functools
import inspect
import json
import os
import stat
import threading
import time
from pathlib import Path

RETURN_CODES = frozenset(
    (
        "ok",
        "timeout",
        "invalid_target",
        "target_unreachable",
        "comm_error",
        "msg_too_big",
        "filter_error",
        "invalid_request",
        "process_exception",
        "authentication_error",
        "service_unavailable",
        "invalid_session",
        "abort_run",
        "unauthenticated",
    )
)
EXCEPTIONS = frozenset(
    (
        "TimeoutError",
        "ConnectionError",
        "RuntimeError",
        "ValueError",
        "AuthenticationError",
        "TargetCellUnreachable",
        "ServiceUnavailable",
        "InvalidSession",
    )
)


class Session:
    def __init__(self, path, targets):
        if set(targets) not in ({"site-1", "site-2"}, {"server", "site-1", "site-2"}):
            raise ValueError("Closed A/B route contract required")
        p = Path(path)
        parent = p.parent.lstat()
        if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) != 0o700:
            raise ValueError("Owned private output directory required")
        self.fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.pid = os.getpid()
        self.targets = frozenset(targets)
        self.lock = threading.Lock()
        self.failed = False
        self.count = 0
        self.request_count = 0
        self.emit(
            {
                "kind": "channel_metadata_start",
                "targets": sorted(targets),
                "payloads_recorded": False,
                "schema": "nvflare-streaming-cc-metadata/v2",
                "recorded_method": "Cell._send_request",
            }
        )

    def emit(self, row):
        if os.getpid() != self.pid:
            return
        with self.lock:
            if self.failed or self.fd is None:
                return
            if self.count >= 1024:
                self.failed = True
                return
            row = dict(row, pid=self.pid, utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            data = (json.dumps(row, sort_keys=True) + "\n").encode()
            try:
                if os.write(self.fd, data) != len(data):
                    self.failed = True
            except OSError:
                self.failed = True
            self.count += 1

    def close(self):
        if os.getpid() != self.pid:
            return
        self.emit({"kind": "channel_metadata_end", "healthy": not self.failed})
        with self.lock:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


TOPICS = frozenset(("get_sites", "request_fresh_token"))
ERROR_LITERALS = {
    "generation_failed": "Failed to generate tokens",
    "generation_handler_exception": "Failed to generate token:",
    "timeout_literal": "timed out",
}


def reply_structure(response, topic, target, required_sites):
    """No token, arbitrary key, site name, route, object text or hash is returned."""
    if response is None:
        return {"payload_kind": "no_response"}
    try:
        payload = response.payload
    except Exception:
        return {"payload_kind": "unreadable"}
    if type(payload) is not dict:
        return {"payload_kind": "other"}
    result = {"payload_kind": "dict"}
    if topic == "request_fresh_token":
        name = payload.get("site_name")
        result["site_name_matches_target"] = type(name) is str and name == target
        infos = payload.get("cc_info")
        result["cc_info_is_list"] = type(infos) is list
        result["cc_info_count"] = len(infos) if type(infos) is list and len(infos) <= 32 else None
        result["cc_info_within_bound"] = type(infos) is list and len(infos) <= 32
    else:
        sites = payload.get("sites")
        valid = type(sites) is list and len(sites) <= 32
        if valid:
            valid = all(
                type(item) in (list, tuple) and len(item) == 2 and all(type(value) is str and value for value in item)
                for item in sites
            )
        result["sites_schema_valid"] = bool(valid)
        result["sites_count"] = len(sites) if type(sites) is list and len(sites) <= 32 else None
        result["required_site_names_present"] = bool(valid and set(required_sites) <= {item[1] for item in sites})
    return result


def install(cell_class, path, targets):
    """Caller must pin source and use a NEW ordinary participant process only."""
    saved = cell_class._send_request
    signature = inspect.signature(saved)
    if list(signature.parameters) != [
        "self",
        "channel",
        "target",
        "topic",
        "request",
        "timeout",
        "secure",
        "optional",
        "abort_signal",
        "progress_wait_cb",
        "num_receivers",
        "receiver_ids",
        "fobs_ctx_props",
    ]:
        raise ValueError("Pinned Cell streaming request signature required")
    defaults = {
        "timeout": 10.0,
        "secure": False,
        "optional": False,
        "abort_signal": None,
        "progress_wait_cb": None,
        "num_receivers": 1,
        "receiver_ids": None,
        "fobs_ctx_props": None,
    }
    if any(
        type(signature.parameters[key].default) is not type(value) or signature.parameters[key].default != value
        for key, value in defaults.items()
    ):
        raise ValueError("Pinned Cell streaming request defaults required")
    session = Session(path, targets)

    @functools.wraps(saved)
    def wrapped(*args, **kwargs):
        try:
            bound = signature.bind(*args, **kwargs).arguments
        except TypeError:
            return saved(*args, **kwargs)
        target = bound.get("target")
        relevant = (
            os.getpid() == session.pid
            and bound.get("channel") == "cc_validation"
            and bound.get("topic") in TOPICS
            and isinstance(target, str)
            and (
                (bound.get("topic") == "get_sites" and target == "server")
                or (bound.get("topic") == "request_fresh_token" and target in session.targets)
            )
        )
        if not relevant:
            return saved(*args, **kwargs)
        topic = bound["topic"]
        started = time.monotonic()
        with session.lock:
            session.request_count += 1
            request_id = session.request_count
        session.emit({"kind": "cc_request_start", "target": target, "topic": topic, "request_id": request_id})
        try:
            response = saved(*args, **kwargs)
        except BaseException as exc:
            name = type(exc).__name__
            session.emit(
                {
                    "kind": "cc_request_exception",
                    "target": target,
                    "topic": topic,
                    "request_id": request_id,
                    "seconds": time.monotonic() - started,
                    "exception_class": name if name in EXCEPTIONS else "other",
                }
            )
            raise
        code = "no_response" if response is None else "unreadable_header"
        has_error = False
        error_categories = []
        if response is not None:
            try:
                candidate = response.get_header("cn__return_code")
                code = candidate if isinstance(candidate, str) and candidate in RETURN_CODES else "unknown_return_code"
                error = response.get_header("cn__error")
                has_error = error is not None
                if type(error) is str and len(error) <= 2048:
                    error_categories = sorted(name for name, literal in ERROR_LITERALS.items() if literal in error)
            except Exception:
                pass
        session.emit(
            {
                "kind": "cc_request_result",
                "target": target,
                "topic": topic,
                "request_id": request_id,
                "seconds": time.monotonic() - started,
                "return_code": code,
                "error_header_present": has_error,
                "error_literal_categories": error_categories,
                "reply_structure": reply_structure(response, topic, target, session.targets),
            }
        )
        return response

    cell_class._send_request = wrapped
    return session, saved

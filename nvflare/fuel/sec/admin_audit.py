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

"""Fail-open administrative auditing with request-local, immutable attribution.

Only approved identifiers and fixed reason codes belong here. Callers must establish
identity, resource ownership and the observed result before recording them. This
module never interprets a command response as evidence of business completion.
"""

import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

from nvflare.fuel.sec.audit import EXCLUDED_ACTIONS, AuditService


@dataclass(frozen=True)
class AdminContext:
    user: str = "unknown"
    request_id: str = ""
    command: str = ""
    job_id: str = ""
    scope_name: str = ""
    identity_source: str = ""


_context = ContextVar("administrative_audit_context", default=AdminContext())
_rejected = ContextVar("administrative_audit_rejected", default=None)
_diagnostic_lock = threading.Lock()
_failed_attempts = 0
_last_warning = float("-inf")
_logger = logging.getLogger(__name__)
_IDENTIFIER = re.compile(r"[A-Za-z0-9_@.%:+\-]{1,256}\Z")
_DETAILS = frozenset(
    {"command", "site", "study", "component", "identity_source", "stage", "status", "operation", "target"}
)
_SENSITIVE = frozenset(
    {
        "delete_job",
        "abort_job",
        "submit_job",
        "clone_job",
        "delete_workspace",
        "app_command",
        "download_job",
        "download_job_components",
        "get_job_log",
        "configure_job_log",
        "configure_site_log",
        "remove_client",
        "disable_client",
        "enable_client",
        "shutdown",
        "restart",
        "register_study",
        "remove_study",
        "add_study_site",
        "remove_study_site",
        "add_study_user",
        "remove_study_user",
        "shell",
        "abort",
        "delete_run",
        "deploy",
        "start",
        "configure_job",
        "delete_runtime",
    }
)


def _failure():
    """Count failed attempts, throttling both normal and fallback diagnostics."""
    global _failed_attempts, _last_warning
    try:
        with _diagnostic_lock:
            _failed_attempts += 1
            now = time.monotonic()
            if now - _last_warning < 60.0:
                return
            _last_warning = now
            attempts = _failed_attempts
        try:
            _logger.warning("Audit recording failed; failed attempts since startup: %d", attempts)
        except Exception:
            try:
                os.write(2, b"Audit recording failed; operation continued.\n")
            except Exception:
                pass
    except Exception:
        # Diagnostics must never change administrative behavior or recurse into audit.
        pass


def _identifier(value):
    if not isinstance(value, str) or (value and not _IDENTIFIER.fullmatch(value)):
        raise ValueError("invalid audit identifier")
    return value


def _actor(value):
    if not isinstance(value, str) or len(value) > 256 or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("invalid audit actor")
    return value or "unknown"


def _scope(value):
    # Privacy scope names are established configuration names, not message tokens.
    # Their existing contract accepts any string; JSON encoding escapes it safely.
    if not isinstance(value, str):
        raise ValueError("invalid audit scope")
    return value


def _approved_context(context):
    if not isinstance(context, AdminContext):
        raise ValueError("invalid audit context")
    return AdminContext(
        user=_actor(context.user),
        request_id=_identifier(context.request_id),
        command=_identifier(context.command),
        job_id=_identifier(context.job_id),
        scope_name=_scope(context.scope_name),
        identity_source=_identifier(context.identity_source),
    )


def snapshot():
    """Return immutable approved values; never retain a connection or session."""
    try:
        return _approved_context(_context.get())
    except Exception:
        _failure()
        return AdminContext()


def update_context(**changes):
    """Safely refine this request's established attribution."""
    try:
        # One unusable optional value must not discard an independently established
        # job/actor/reference, or retain stale scope from a different job.
        validators = {"user": _actor, "scope_name": _scope}
        defaults = AdminContext()
        approved = {}
        failed = False
        for key, value in changes.items():
            if key not in defaults.__dataclass_fields__:
                raise ValueError("unknown audit context field")
            try:
                approved[key] = validators.get(key, _identifier)(value)
            except Exception:
                approved[key] = getattr(defaults, key)
                failed = True
        context = _approved_context(replace(_context.get(), **approved))
        _context.set(context)
        if failed:
            _failure()
        return context
    except Exception:
        _failure()
        return snapshot()


@contextmanager
def audit_context(context=None, **changes):
    """Isolate a request and restore its caller's context on every exit."""
    try:
        prepared = _approved_context(replace(snapshot() if context is None else context, **changes))
    except Exception:
        _failure()
        prepared = AdminContext()
    token = _context.set(prepared)
    rejection_token = _rejected.set(False)
    try:
        yield prepared
    finally:
        _rejected.reset(rejection_token)
        _context.reset(token)


def is_sensitive(command):
    """Extra handling records are limited to explicit administrative operations."""
    try:
        return isinstance(command, str) and command.rsplit(".", 1)[-1] in _SENSITIVE
    except Exception:
        return False


def record(
    action, *, context=None, user=None, ref=None, job_id=None, scope_name=None, reason="", writer=None, **details
):
    """Write one administrative record or return an empty identifier on failure.

    The writer's ordinary failure semantics are intentionally unchanged for callers
    outside this helper. An empty return can mean intentional simulator silence or
    an excluded poll; otherwise it is a failed recording attempt.
    """
    try:
        action = _identifier(action)
        if action in EXCLUDED_ACTIONS:
            return ""
        auditor = AuditService.get_auditor() if writer is None else writer
        if auditor is None:
            raise RuntimeError("audit writer unavailable")
        if getattr(auditor, "_admin_audit_noop", False) is True:
            return ""
        approved = _approved_context(_context.get() if context is None else context)
        actor = _actor(approved.user if user is None else user)
        reference = _identifier(approved.request_id if ref is None else ref)
        job = _identifier(approved.job_id if job_id is None else job_id)
        scope = _scope(approved.scope_name if scope_name is None else scope_name)
        message = []
        if reason:
            message.append("reason=" + _identifier(reason))
        if approved.command and "command" not in details:
            details["command"] = approved.command
        if approved.identity_source and "identity_source" not in details:
            details["identity_source"] = approved.identity_source
        for key, value in details.items():
            if key not in _DETAILS:
                raise ValueError("unapproved audit detail")
            if value:
                message.append(key + "=" + _identifier(value))
        kwargs = {"user": actor, "action": action, "ref": reference, "msg": " ".join(message)}
        # Preserve existing subclasses' four-argument method for ordinary events.
        if job or scope:
            kwargs.update(job_id=job, scope_name=scope)
        result = auditor.add_event(**kwargs)
        if not isinstance(result, str) or not result:
            raise RuntimeError("audit writer returned no identifier")
        return result
    except Exception:
        _failure()
        return ""


def reject(action, **kwargs):
    """Record a request's rejection once, including fallback preprocessing checks."""
    try:
        if _rejected.get() is True:
            return ""
        if _rejected.get() is not None:
            _rejected.set(True)
        return record(action, **kwargs)
    except Exception:
        _failure()
        return ""


def recording_failed():
    """Report a failure in call-site audit-only preparation, without auditing it."""
    _failure()


def prepare(callback, *, default=None):
    """Protect audit-only preparation; never use this to wrap the operation itself."""
    try:
        return callback()
    except Exception:
        _failure()
        return default

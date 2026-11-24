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

"""Protected task/workflow observations, with bounded collection outside locks.

Only locally established job/scope and explicitly approved task identity belong
in a context. Payloads, exception text, credentials and unchecked peer values do
not. Existing legacy task audit calls and their transport headers are unchanged.
"""

import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
from functools import wraps

from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.shareable import ReservedHeaderKey
from nvflare.fuel.sec.admin_audit import recording_failed
from nvflare.fuel.sec.audit import AuditService


@dataclass(frozen=True)
class TaskAuditContext:
    job_id: str = ""
    scope_name: str = ""
    task_name: str = ""
    task_id: str = ""
    workflow_id: str = ""
    attempt_id: str = ""
    site: str = ""
    reference: str = ""


@dataclass(frozen=True)
class _Fact:
    action: str
    context: TaskAuditContext
    reason: str
    status: str
    stage: str


class _Pending:
    def __init__(self):
        self.facts = []
        self.failures = 0


_context = ContextVar("task_audit_context", default=TaskAuditContext())
_pending = ContextVar("task_audit_pending", default=None)
_writing = ContextVar("task_audit_writing", default=False)
_CURRENT = object()
_MAX_PENDING = 1024
_TOKEN = re.compile(r"[a-zA-Z0-9_.:+-]{1,128}\Z")
_RETURN_CODES = frozenset(v for k, v in vars(ReturnCode).items() if k.isupper() and isinstance(v, str))


def _failure():
    """Defer diagnostic I/O as well as audit I/O while an operation owns locks."""
    try:
        pending = _pending.get()
        if pending is not None:
            pending.failures += 1
        else:
            recording_failed()
    except Exception:
        pass


def prepare(callback, *, default=None):
    """Protect audit-only preparation; never wrap the operation being audited."""
    try:
        return callback()
    except Exception:
        _failure()
        return default


def _approved(context):
    if not isinstance(context, TaskAuditContext):
        raise ValueError("invalid task audit context")
    values = {field.name: getattr(context, field.name) for field in fields(TaskAuditContext)}
    if any(not isinstance(value, str) for value in values.values()):
        raise ValueError("invalid task audit context value")
    return TaskAuditContext(**values)


def build_context(fl_ctx=None, **overrides):
    """Copy local job/scope only; task identity and references require approval.

    Return None on a preparation failure. Explicit None passed to record is an
    invalid context, rather than an invitation to inherit another request.
    """
    try:
        values = {}
        if fl_ctx is not None:
            if "job_id" not in overrides:
                values["job_id"] = fl_ctx.get_job_id() or ""
            if "scope_name" not in overrides:
                values["scope_name"] = fl_ctx.get_prop(FLContextKey.EFFECTIVE_JOB_SCOPE_NAME, "") or ""
        values.update({key: "" if value is None else value for key, value in overrides.items()})
        return _approved(TaskAuditContext(**values))
    except Exception:
        _failure()
        return None


def capture_operation_context(fl_ctx):
    """Snapshot job and scope with a matching, approved local reference."""

    def capture():
        local = build_context(fl_ctx)
        if local is None:
            return None
        parent = current_context()
        reference = ""
        if (
            parent is not None
            and local.job_id
            and local.job_id == parent.job_id
            and local.scope_name == parent.scope_name
        ):
            reference = parent.reference
        return TaskAuditContext(job_id=local.job_id, scope_name=local.scope_name, reference=reference)

    return prepare(capture)


def current_context():
    return _context.get()


def set_context(context):
    """Refine a context inside use_context; that outer scope owns resetting it."""
    approved = prepare(lambda: _approved(context))
    _context.set(approved)
    return approved


@contextmanager
def use_context(context):
    approved = prepare(lambda: _approved(context))
    token = _context.set(approved)
    try:
        yield approved
    finally:
        _context.reset(token)


def code_status(code):
    """Retain only built-in return codes; application-defined values may be private."""
    try:
        if code is None:
            return "unspecified"
        return code if isinstance(code, str) and code in _RETURN_CODES else "custom_return_code"
    except Exception:
        _failure()
        return "custom_return_code"


def reply_status(reply):
    """Report the explicit return code without invoking operational defaults or overrides."""

    def read():
        headers = dict.get(reply, ReservedHeaderKey.HEADERS)
        return code_status(dict.get(headers, ReservedHeaderKey.RC) if headers is not None else None)

    return prepare(read, default="unavailable")


def _token(value):
    if not isinstance(value, str) or (value and not _TOKEN.fullmatch(value)):
        raise ValueError("invalid fixed task audit token")
    return value


def _write(fact):
    if _writing.get():
        _failure()
        return ""
    token = _writing.set(True)
    try:
        auditor = AuditService.get_auditor()
        if auditor is None:
            raise RuntimeError("audit writer unavailable")
        if getattr(auditor, "_admin_audit_noop", False) is True:
            return ""
        context = fact.context
        details = []
        for key, value in (("reason", fact.reason), ("status", fact.status), ("stage", fact.stage)):
            if value:
                details.append(key + "=" + value)
        # Quoting approved names preserves spaces, Unicode and line breaks in
        # existing task/workflow names without introducing another record line.
        for key, value in (
            ("task", context.task_name),
            ("task_id", context.task_id),
            ("workflow", context.workflow_id),
            ("attempt_id", context.attempt_id),
            ("site", context.site),
        ):
            if value:
                details.append(key + "=" + json.dumps(value, ensure_ascii=True))
        kwargs = {"user": "unknown", "action": fact.action, "ref": context.reference, "msg": " ".join(details)}
        if context.job_id or context.scope_name:
            kwargs.update(job_id=context.job_id, scope_name=context.scope_name)
        event_id = auditor.add_event(**kwargs)
        if not isinstance(event_id, str) or not event_id:
            raise RuntimeError("audit writer returned no identifier")
        return event_id
    except Exception:
        _failure()
        return ""
    finally:
        _writing.reset(token)


def record(action, *, context=_CURRENT, reason="", status="", stage=""):
    """Record approved facts, returning no fabricated identifier while deferred."""
    try:
        if _writing.get():
            _failure()
            return ""
        fact = _Fact(
            _token(action),
            _approved(current_context() if context is _CURRENT else context),
            _token(reason),
            _token(status),
            _token(stage),
        )
        if not fact.action:
            raise ValueError("missing task audit action")
        pending = _pending.get()
        if pending is not None:
            if len(pending.facts) >= _MAX_PENDING:
                pending.failures += 1
            else:
                pending.facts.append(fact)
            return ""
        return _write(fact)
    except Exception:
        _failure()
        return ""


@contextmanager
def collect():
    """Flush only at the outermost operation, after its owned locks are released.

    Never wrap a whole workflow lifetime. Deferred timestamps are write times,
    and do not establish a total observation order across concurrent operations.
    """
    if _pending.get() is not None:
        yield
        return
    pending = _Pending()
    token = _pending.set(pending)
    try:
        yield
    finally:
        # Detach before calling writers/loggers: those may reenter instrumentation.
        facts = tuple(pending.facts)
        failures = pending.failures
        _pending.reset(token)
        for fact in facts:
            _write(fact)
        for _ in range(failures):
            _failure()


def collect_events(method):
    """Wrap a lock-owning operation; nested wrappers share one bounded collector."""

    @wraps(method)
    def wrapped(*args, **kwargs):
        with collect():
            return method(*args, **kwargs)

    return wrapped

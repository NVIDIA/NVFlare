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

"""Record training model observations without retaining model data."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

from nvflare.fuel.sec import task_audit

_implementation = ContextVar("model_audit_implementation", default=None)
_static_save = ContextVar("model_audit_static_save", default=None)
_SAVE_OPERATIONS = ("save", "best_save")


def _capture_context(fl_ctx):
    local = task_audit.build_context(fl_ctx)
    if local is None:
        return None
    parent = task_audit.current_context()
    reference = ""
    if parent is not None and local.job_id and local.job_id == parent.job_id and local.scope_name == parent.scope_name:
        reference = parent.reference
    return task_audit.TaskAuditContext(job_id=local.job_id, scope_name=local.scope_name, reference=reference)


class _Operation:
    def __init__(self, name, context, owner):
        self.name = name
        self.context = context
        self.owner_id = id(owner) if owner is not None else None
        self.error_stage = ""

    def record(self, outcome, *, stage="", reason="", status=""):
        # Context preparation already reported a failure if no snapshot was made.
        if self.context is not None:
            task_audit.record(
                f"model.{self.name}.{outcome}", context=self.context, stage=stage, reason=reason, status=status
            )

    @contextmanager
    def stage(self, name, *, reason=""):
        scope = _use_implementation(self) if name == "implementation" else nullcontext()
        try:
            with scope:
                yield
        except Exception:
            self.error_stage = name
            self.record("stage_error", stage=name, reason=reason)
            raise
        else:
            self.record("stage_returned", stage=name, reason=reason)

    def call(self, stage, function, /, *args, **kwargs):
        with self.stage(stage):
            return function(*args, **kwargs)

    def returned(self, value=None):
        status = ""
        if self.name == "load":
            status = "no_value" if value is None else "value_returned"
        self.record("returned", status=status)
        return value

    def skipped(self, *, reason):
        self.record("skipped", reason=reason)


@contextmanager
def operation(name, fl_ctx, *, owner=None):
    """Collect one short operation, joining any outer lock owner's collector."""
    with task_audit.collect(), suspend_implementation():
        context = task_audit.prepare(lambda: _capture_context(fl_ctx))
        audit = _Operation(name, context, owner)
        try:
            audit.record("started")
            yield audit
        except Exception:
            audit.record("error", stage=audit.error_stage)
            raise


def suspend_implementation():
    """Keep unrelated work from contributing stages to an outer model operation."""
    return _use_implementation(None)


@contextmanager
def _use_implementation(audit):
    token = _implementation.set(audit)
    try:
        yield
    finally:
        _implementation.reset(token)


def _owned_implementation(owner, operations):
    audit = _implementation.get()
    if audit is not None and owner is not None and audit.owner_id == id(owner) and audit.name in operations:
        return audit
    return None


def saving(owner):
    return _owned_implementation(owner, _SAVE_OPERATIONS) is not None


def load_step(owner, stage, function, /, *args, **kwargs):
    return _storage_step(_owned_implementation(owner, ("load",)), stage, function, *args, **kwargs)


def save_step(owner, stage, function, /, *args, **kwargs):
    return _storage_step(_owned_implementation(owner, _SAVE_OPERATIONS), stage, function, *args, **kwargs)


@contextmanager
def static_save_step(owner):
    """Allow a static file helper to observe a step of the calling owner's save."""
    token = _static_save.set(_owned_implementation(owner, _SAVE_OPERATIONS))
    try:
        yield
    finally:
        _static_save.reset(token)


def static_save_call(stage, function, /, *args, **kwargs):
    audit = _static_save.get()
    if audit is not _implementation.get():
        audit = None
    return _storage_step(audit, stage, function, *args, **kwargs)


def _storage_step(audit, stage, function, /, *args, **kwargs):
    """Observe the specific call, keeping any enclosing error handling unchanged."""
    if audit is None:
        return function(*args, **kwargs)
    try:
        result = function(*args, **kwargs)
    except Exception:
        audit.record("stage_error", stage=stage)
        raise
    audit.record("stage_returned", stage=stage)
    return result

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
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

from nvflare.apis.fl_constant import FLContextKey
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.abstract.persistor_filter import PersistorFilter
from nvflare.fuel.sec import admin_audit, task_audit
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.private.fed.simulator.simulator_audit import SimulatorAuditor

_ALLOWED_FIELDS = {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}


class LocalContext:
    def __init__(self, job="job-a", scope="approved scope", model_filter=None):
        self.job = job
        self.scope = scope
        self.model_filter = model_filter

    def get_job_id(self):
        return self.job

    def get_prop(self, key, default):
        assert key == FLContextKey.EFFECTIVE_JOB_SCOPE_NAME
        return self.scope

    def get_engine(self):
        return SimpleNamespace(get_component=lambda name: self.model_filter)


class RecordingPersistor(ModelPersistor):
    def __init__(self, model, filter_id=None, load=None, save=None):
        super().__init__(filter_id)
        self.model = model
        self.load_hook = load
        self.save_hook = save
        self.saved = []

    def load_model(self, fl_ctx):
        if self.load_hook is not None:
            self.load_hook(fl_ctx)
        return self.model

    def save_model(self, model, fl_ctx):
        self.saved.append(model)
        if self.save_hook is not None:
            self.save_hook(fl_ctx)


class Transform(PersistorFilter):
    def __init__(self, calls, transformed, error, fail_at=None):
        super().__init__()
        self.calls = calls
        self.transformed = transformed
        self.error = error
        self.fail_at = fail_at

    def _step(self, name, learnable):
        self.calls.append((name, learnable))
        if self.fail_at == name:
            raise self.error
        return self.transformed

    def process_pre_save(self, learnable, fl_ctx):
        return self._step("pre_save_filter", learnable)

    def process_post_save(self, learnable, fl_ctx):
        return self._step("post_save_filter", learnable)

    def process_post_load(self, learnable, fl_ctx):
        return self._step("post_load_filter", learnable)

    def process_post_get(self, learnable, fl_ctx):
        return learnable


@pytest.fixture
def writer(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    auditor = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", auditor)
    monkeypatch.setattr(admin_audit, "_failed_attempts", 0)
    monkeypatch.setattr(admin_audit, "_last_warning", float("-inf"))
    yield auditor, path
    auditor.close()


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _facts(path):
    return [(item["action"], item.get("message", "")) for item in _records(path)]


def test_model_facts_keep_only_verified_entry_context(writer):
    auditor, path = writer
    parent_id = auditor.add_event(
        "unknown", "workflow.execution.initialized", job_id="job-a", scope_name="approved scope"
    )
    parent = task_audit.build_context(
        job_id="job-a",
        scope_name="approved scope",
        reference=parent_id,
        task_name="private-task",
        site="private-peer",
    )
    source = LocalContext()
    model = {"private-weight": 123, "metadata": {"private-metadata": True}}

    def mutate_context(ctx):
        ctx.job = "job-b"
        ctx.scope = "wrong scope"

    persistor = RecordingPersistor(model, load=mutate_context)
    with task_audit.use_context(parent):
        assert persistor.load(source) is model
        assert task_audit.current_context() == parent
        assert persistor.load(source) is model
    assert RecordingPersistor(None).load(LocalContext("", "")) is None

    records = _records(path)[1:]
    assert len(records) == 9
    assert [r["action"] for r in records[:3]] == [
        "model.load.started",
        "model.load.stage_returned",
        "model.load.returned",
    ]
    assert [r.get("message") for r in records[:3]] == [None, "stage=implementation", "status=value_returned"]
    assert all(
        r["job"] == "job-a" and r["scope"] == "approved scope" and r["reference"] == parent_id for r in records[:3]
    )
    assert all(r["job"] == "job-b" and r["scope"] == "wrong scope" and "reference" not in r for r in records[3:6])
    assert all(not ({"reference", "job", "scope"} & r.keys()) for r in records[6:])
    assert records[-1]["message"] == "status=no_value"
    for r in records:
        assert set(r) <= _ALLOWED_FIELDS
        assert all(isinstance(v, str) for v in r.values())
        assert r["user"] == "unknown"
        assert r["timestamp"].endswith("+00:00")
    for marker in ("private-weight", "private-metadata", "private-task", "private-peer"):
        assert marker not in path.read_text()


@pytest.mark.parametrize("fail_at", [None, "pre_save_filter", "implementation", "post_save_filter"])
def test_save_stages_preserve_transforms_and_applied_effects(writer, fail_at):
    _, path = writer
    original, transformed = object(), object()
    calls = []
    error = ValueError("secret-value-and-private/path")
    transform = Transform(calls, transformed, error, fail_at)

    def save(ctx):
        calls.append(("implementation", ctx))
        if fail_at == "implementation":
            raise error

    source = LocalContext(model_filter=transform)
    persistor = RecordingPersistor(None, "private-filter-identifier", save=save)
    expected = [("pre_save_filter", original), ("implementation", source), ("post_save_filter", transformed)]
    stages = [s for s, _ in expected]
    if fail_at is None:
        assert persistor.save(original, source) is None
        outcome = ("model.save.returned", "")
    else:
        with pytest.raises(ValueError) as raised:
            persistor.save(original, source)
        assert raised.value is error
        end = stages.index(fail_at) + 1
        expected, stages = expected[:end], stages[:end]
        outcome = ("model.save.error", f"stage={fail_at}")
    assert calls == expected
    assert persistor.saved == ([] if fail_at == "pre_save_filter" else [transformed])
    assert _facts(path) == [
        ("model.save.started", ""),
        *(
            ("model.save.stage_error" if stage == fail_at else "model.save.stage_returned", f"stage={stage}")
            for stage in stages
        ),
        outcome,
    ]
    assert "secret-value" not in path.read_text()
    assert "private-filter-identifier" not in path.read_text()


@pytest.mark.parametrize("mode", ["returned", "post_load_filter", "invalid_filter", "implementation"])
def test_load_keeps_original_return_or_raised_error(writer, mode):
    _, path = writer
    original, transformed = object(), object()
    error = RuntimeError("private load error")
    calls = []
    transform = Transform(calls, transformed, error, mode)

    def load(_):
        if mode == "implementation":
            raise error

    source = LocalContext(model_filter=None if mode == "invalid_filter" else transform)
    persistor = RecordingPersistor(original, "filter", load=load)
    if mode == "returned":
        assert persistor.load(source) is transformed
        assert calls == [("post_load_filter", original)]
        assert _facts(path)[-1] == ("model.load.returned", "status=value_returned")
    else:
        with pytest.raises(ValueError if mode == "invalid_filter" else RuntimeError) as raised:
            persistor.load(source)
        if mode != "invalid_filter":
            assert raised.value is error
        stage = "implementation" if mode == "implementation" else "post_load_filter"
        assert _facts(path)[-2:] == [
            ("model.load.stage_error", f"stage={stage}"),
            ("model.load.error", f"stage={stage}"),
        ]
        assert "private load error" not in path.read_text()
    assert [action for action, _ in _facts(path)].count("model.load.started") == 1


@pytest.mark.parametrize("state", ["missing", "closed", "empty", "failing", "simulator", "invalid_context"])
def test_recording_failure_preserves_model_outcome_and_does_not_retry(writer, monkeypatch, state, capfd):
    auditor, path = writer
    ctx = LocalContext()
    model = object()

    def fail(**kwargs):
        raise OSError("private writer failure")

    if state == "closed":
        auditor.close()
    elif state == "invalid_context":

        def fail_context():
            raise ValueError("private context failure")

        ctx.get_job_id = fail_context
    else:
        replacement = {
            "missing": None,
            "empty": SimpleNamespace(add_event=lambda **kwargs: ""),
            "failing": SimpleNamespace(add_event=fail),
            "simulator": SimulatorAuditor(),
        }[state]
        monkeypatch.setattr(AuditService, "the_auditor", replacement)
    persistor = RecordingPersistor(model)
    assert persistor.load(ctx) is model
    assert persistor.save(model, ctx) is None
    assert persistor.saved == [model]
    expected = {"simulator": 0, "invalid_context": 2}.get(state, 6)
    assert admin_audit._failed_attempts == expected
    diagnostic = "Audit recording failed; failed attempts since startup: 1\n"
    assert capfd.readouterr().err == (diagnostic if expected else "")
    assert _records(path) == []


def test_parent_context_is_local_to_a_job_and_scope_in_parallel(writer):
    auditor, path = writer
    barrier = Barrier(2)
    references = {}
    for job in ("one", "two"):
        references[job] = auditor.add_event("unknown", "workflow.execution.initialized", job_id=job, scope_name="scope")

    def load(job):
        parent = task_audit.build_context(job_id=job, scope_name="scope", reference=references[job])
        with task_audit.use_context(parent):
            model = RecordingPersistor(None, load=lambda _: barrier.wait(timeout=5))
            model.load(LocalContext(job, "scope"))
            RecordingPersistor(None).load(LocalContext(job, "other scope"))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(load, job) for job in references]
        for future in futures:
            future.result(timeout=10)

    records = [r for r in _records(path) if r["action"].startswith("model.")]
    assert len(records) == 12
    for record in records:
        if record["scope"] == "scope":
            assert record["reference"] == references[record["job"]]
        else:
            assert record["scope"] == "other scope"
            assert "reference" not in record

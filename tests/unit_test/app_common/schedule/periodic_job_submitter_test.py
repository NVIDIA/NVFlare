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

import io
import zipfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.schedule import periodic_job_submitter as module
from nvflare.app_common.schedule.periodic_job_submitter import SIGNER_DIR, PeriodicJobSubmitter
from nvflare.app_opt.job_launcher.workspace_cell_transfer import _zip_workspace_to_file
from nvflare.fuel.hci.security import IdentityKey
from nvflare.security.study_registry import StudyRegistryService

T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
RECIPE = "nvflare.recipe.catalog.CatalogRecipe"


class FakeJobManager:
    def __init__(self):
        self.jobs, self.zips, self.fail = {}, {}, False

    def create(self, meta, content, fl_ctx):
        if self.fail:
            raise RuntimeError("store down")
        job_id = f"job-{len(self.jobs) + 1}"
        meta.update(job_id=job_id, status="SUBMITTED", submit_time=T0.timestamp())
        self.jobs[job_id], self.zips[job_id] = meta, content
        return meta

    def get_job(self, job_id, fl_ctx):
        return SimpleNamespace(meta=self.jobs[job_id]) if job_id in self.jobs else None


class FakeRunner:
    """The job runner's abort-before-start: it reports STARTING while it is starting the job."""

    STARTING = object()

    def __init__(self, jobs):
        self.jobs, self.starting, self.aborted = jobs, set(), []

    def abort_before_start(self, job_id, fl_ctx):
        if job_id in self.starting:
            return self.STARTING
        status = self.jobs.jobs[job_id]["status"]
        self.jobs.jobs[job_id]["status"] = "FINISHED:ABORTED"
        self.aborted.append(job_id)
        return status


class FakeEngine:
    def __init__(self, jobs, block_reason=None):
        self.jobs, self.job_runner, self.block_reason, self.events = jobs, FakeRunner(jobs), block_reason, []

    def get_component(self, name):
        return self.jobs

    def new_context(self):
        return FLContext()

    def fire_event(self, event_type, fl_ctx):
        self.events.append((event_type, fl_ctx.get_prop(FLContextKey.JOB_META)["study"]))
        if self.block_reason:
            fl_ctx.set_prop(FLContextKey.JOB_BLOCK_REASON, self.block_reason, private=True, sticky=False)


def make(studies=("study1", "study2"), **kw):
    jobs = FakeJobManager()
    submitter = PeriodicJobSubmitter(
        RECIPE, interval_s=3600, studies=list(studies), jitter_fraction=0.0, startup_delay_s=0, **kw
    )
    submitter._engine = FakeEngine(jobs)
    submitter._now = lambda: submitter.clock
    submitter.clock = T0
    submitter.jobs = jobs
    return submitter


def finish(submitter, job_id):
    submitter.jobs.jobs[job_id]["status"] = "FINISHED:COMPLETED"


class TestSchedule:
    def test_each_study_gets_one_job_with_the_recipes_definition_and_the_service_identity(self):
        s = make()
        s._tick()
        assert [m["study"] for m in s.jobs.jobs.values()] == ["study1", "study2"]
        meta = s.jobs.jobs["job-1"]
        assert meta["submitter_name"] == "scheduled-job-submitter" and meta["fdn_catalog_read"] is True
        assert meta["deploy_map"] == {"app": ["@ALL"]}
        names = zipfile.ZipFile(io.BytesIO(s.jobs.zips["job-1"])).namelist()
        assert "catalog_read/" in names and "catalog_read/meta.json" in names

    def test_a_study_is_not_resubmitted_while_its_job_runs(self):
        s = make(submit_timeout_s=3 * 3600)
        s._tick()
        s.clock += timedelta(hours=2)
        s._tick()
        assert len(s.jobs.jobs) == 2

    def test_the_next_job_is_due_an_interval_after_the_last_and_only_when_the_last_finished(self):
        s = make(studies=("study1",))
        s._tick()
        finish(s, "job-1")
        s.clock += timedelta(minutes=30)
        s._tick()
        assert len(s.jobs.jobs) == 1  # not due yet
        s.clock += timedelta(minutes=31)
        s._tick()
        assert len(s.jobs.jobs) == 2

    def test_the_first_job_waits_for_the_startup_delay(self):
        s = make(studies=("study1",))
        s._startup_delay_s = 60
        s._tick()
        assert s._next["study1"] <= T0 + timedelta(seconds=60)

    def test_a_failed_submit_retries_soon_and_success_restores_the_normal_interval(self):
        s = make(studies=("study1",))
        s.jobs.fail = True
        s._tick()
        s.jobs.fail = False
        s._tick()
        assert s.jobs.jobs == {}
        s.clock += timedelta(seconds=29)
        s._tick()
        assert s.jobs.jobs == {}
        s.clock += timedelta(seconds=1)
        s._tick()
        assert len(s.jobs.jobs) == 1
        assert s._next["study1"] == s.clock + timedelta(hours=1)

    def test_a_failed_study_does_not_delay_other_studies(self, monkeypatch):
        s = make()
        submit = s._submit

        def submit_one(study):
            if study == "study1":
                raise ValueError("temporary overlap")
            return submit(study)

        monkeypatch.setattr(s, "_submit", submit_one)
        s._tick()
        assert set(s._running) == {"study2"}
        assert s._next["study1"] == T0 + timedelta(seconds=30)

    def test_a_job_that_never_started_is_aborted_so_the_study_is_not_stuck(self):
        s = make(studies=("study1",), submit_timeout_s=600)
        s._tick()
        s.jobs.jobs["job-1"]["submit_time"] = (T0 - timedelta(hours=1)).timestamp()
        s._reap()
        assert s._engine.job_runner.aborted == ["job-1"] and "study1" not in s._running
        assert "job-1" in s.jobs.jobs  # aborted through the job runner, never deleted under it

    def test_a_job_the_runner_is_starting_is_left_to_run(self):
        s = make(studies=("study1",), submit_timeout_s=600)
        s._tick()
        s.jobs.jobs["job-1"]["submit_time"] = (T0 - timedelta(hours=1)).timestamp()
        s._engine.job_runner.starting.add("job-1")
        s._reap()
        assert s._engine.job_runner.aborted == [] and s._running == {"study1": "job-1"}

    def test_every_job_passes_the_submit_job_event(self):
        s = make()
        s._tick()
        assert s._engine.events == [(EventType.SUBMIT_JOB, "study1"), (EventType.SUBMIT_JOB, "study2")]

    def test_a_job_a_component_blocks_is_not_stored(self):
        s = make(studies=("study1",))
        s._engine.block_reason = "BYOC job not allowed for CC"
        s._tick()
        assert s.jobs.jobs == {} and s._running == {}

    def test_a_deleted_job_is_forgotten(self):
        s = make(studies=("study1",))
        s._tick()
        del s.jobs.jobs["job-1"]
        s._reap()
        assert s._running == {}

    def test_a_running_job_is_left_alone(self):
        s = make(studies=("study1",), submit_timeout_s=1)
        s._tick()
        s.jobs.jobs["job-1"]["status"] = "RUNNING"
        s.jobs.jobs["job-1"]["submit_time"] = 0.0
        s._reap()
        assert s._engine.job_runner.aborted == [] and s._running == {"study1": "job-1"}

    def test_a_recipe_that_cannot_be_built_is_a_failed_submit_not_a_crash(self):
        s = make(studies=("study1",))
        s._recipe_args = {"datasets": []}
        s._tick()
        assert s.jobs.jobs == {} and s._running == {}


class TestStudies:
    def test_the_configured_studies_win(self):
        assert make(studies=("a",)).studies == ["a"]

    def test_every_registered_study_by_default_and_the_default_study_without_a_registry(self, monkeypatch):
        s = PeriodicJobSubmitter(RECIPE)
        monkeypatch.setattr(StudyRegistryService, "get_registry", staticmethod(lambda: None))
        assert s.studies == ["default"]
        registry = SimpleNamespace(get_studies=lambda: ["b", "a"])
        monkeypatch.setattr(StudyRegistryService, "get_registry", staticmethod(lambda: registry))
        assert s.studies == ["a", "b"]


class TestSigning:
    def test_parent_only_credentials_are_loaded_and_excluded_from_the_real_job_bundle(self, tmp_path, monkeypatch):
        private = tmp_path / SIGNER_DIR
        private.mkdir(parents=True)
        key, cert = private / "admin.key", private / "admin.crt"
        key.write_text("SYNTHETIC_PRIVATE_KEY")
        cert.write_text("SYNTHETIC_CERT")
        (tmp_path / "local").mkdir()
        (tmp_path / "local" / "resources.json").write_text("{}")
        monkeypatch.setattr(module, "load_crt", lambda path: path)
        monkeypatch.setattr(module, "cert_to_dict", lambda cert: cert)
        monkeypatch.setattr(
            module,
            "get_identity_info",
            lambda cert: {IdentityKey.NAME: "service", IdentityKey.ORG: "org", IdentityKey.ROLE: "project_admin"},
        )
        s = make(signer_key=str(SIGNER_DIR / "admin.key"), signer_cert=str(SIGNER_DIR / "admin.crt"))
        s._load_signer(SimpleNamespace(get_root_dir=lambda: str(tmp_path)))
        assert s._signer == (str(key), str(cert)) and s._submitter == ("service", "org", "project_admin")
        bundle = tmp_path / "bundle.zip"
        _zip_workspace_to_file(str(tmp_path), "j1", str(bundle))
        with zipfile.ZipFile(bundle) as zf:
            assert zf.namelist() == ["local/resources.json"]
            assert all(b"SYNTHETIC_PRIVATE_KEY" not in zf.read(name) for name in zf.namelist())

    @pytest.mark.parametrize(
        "location",
        ["local/admin.key", "startup/admin.key", "j1/admin.key", "private/scheduled_jobs/../../local/admin.key"],
    )
    def test_credentials_in_job_visible_directories_are_refused(self, tmp_path, location):
        s = make(signer_key=location, signer_cert=str(SIGNER_DIR / "admin.crt"))
        with pytest.raises(ValueError, match="private/scheduled_jobs"):
            s._load_signer(SimpleNamespace(get_root_dir=lambda: str(tmp_path)))
        assert s._signer is None

    def test_a_symlink_from_the_private_directory_to_a_job_visible_key_is_refused(self, tmp_path):
        private = tmp_path / SIGNER_DIR
        private.mkdir(parents=True)
        local = tmp_path / "local"
        local.mkdir()
        (local / "admin.key").write_text("key")
        (private / "admin.key").symlink_to(local / "admin.key")
        s = make(signer_key=str(SIGNER_DIR / "admin.key"), signer_cert=str(SIGNER_DIR / "admin.crt"))
        with pytest.raises(ValueError, match="private/scheduled_jobs"):
            s._load_signer(SimpleNamespace(get_root_dir=lambda: str(tmp_path)))

    @pytest.mark.parametrize(
        "key,cert", [("private/scheduled_jobs/admin.key", None), (None, "private/scheduled_jobs/admin.crt")]
    )
    def test_a_partial_signer_configuration_is_refused(self, tmp_path, key, cert):
        s = make(signer_key=key, signer_cert=cert)
        with pytest.raises(ValueError, match="set both"):
            s._load_signer(SimpleNamespace(get_root_dir=lambda: str(tmp_path)))

    def test_a_job_is_signed_with_the_service_identity_before_it_is_zipped(self, monkeypatch):
        calls = []
        monkeypatch.setattr(module, "load_private_key_file", lambda path: f"key:{path}")
        monkeypatch.setattr(module, "sign_folders", lambda folder, key, cert: calls.append((folder, key, cert)))
        s = make(studies=("study1",))
        s._signer = ("/ws/k.key", "/ws/k.crt")
        s._tick()
        assert (
            len(calls) == 1 and calls[0][0].endswith("catalog_read") and calls[0][1:] == ("key:/ws/k.key", "/ws/k.crt")
        )

    def test_no_signer_means_an_unsigned_job(self, monkeypatch):
        monkeypatch.setattr(module, "sign_folders", lambda *a: pytest.fail("must not sign"))
        s = make(studies=("study1",))
        s._tick()
        assert len(s.jobs.jobs) == 1

    @pytest.mark.parametrize("required", [True, False])
    def test_missing_signer_on_a_server_that_requires_signing_is_reported(self, monkeypatch, required):
        monkeypatch.setattr(module, "require_signed_jobs", lambda workspace: required)
        s = make()
        errors = []
        monkeypatch.setattr(s.logger, "error", errors.append)
        s._load_signer(SimpleNamespace(get_root_dir=lambda: "/ws"))
        assert bool(errors) is required and s._signer is None


def test_invalid_startup_signer_is_explicitly_reported_and_never_starts_unsigned(monkeypatch, caplog):
    s = make(signer_key="half-configured")
    ctx = FLContext()
    monkeypatch.setattr(ctx, "get_engine", lambda: s._engine)
    monkeypatch.setattr(ctx, "get_workspace", lambda: None)
    started = []
    monkeypatch.setattr(module.threading, "Thread", lambda **kw: started.append(kw))
    s.handle_event(EventType.SYSTEM_START, ctx)
    assert not started and s._signer is None
    assert "periodic scheduling is disabled: invalid signer configuration" in caplog.text


def test_unreadable_running_job_does_not_starve_other_studies(monkeypatch):
    s = make()
    s._tick()
    first, second = s._running["study1"], s._running["study2"]
    finish(s, second)
    original = s.jobs.get_job

    def read(job_id, ctx):
        if job_id == first:
            raise OSError("unreadable record")
        return original(job_id, ctx)

    monkeypatch.setattr(s.jobs, "get_job", read)
    s.clock += timedelta(hours=1)
    s._tick()
    assert s._running["study1"] == first
    assert s._running["study2"] != second
    assert len(s.jobs.jobs) == 3

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

"""A server component that submits the job of a recipe on a schedule, once per Study.

A deployment that requires signed jobs accepts a job only with the signature of an admin certificate that chains to
the root CA, exactly as for a job an admin client submits. The submitter therefore signs each job with a service
identity the operator provisions: an admin-role key and certificate it can read.
"""

import io
import os
import random
import tempfile
import threading
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, SystemComponents
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import DEFAULT_STUDY, JobMetaKey, RunStatus
from nvflare.fuel.hci.security import IdentityKey, get_identity_info
from nvflare.fuel.utils.class_loader import load_class
from nvflare.lighter.utils import cert_to_dict, load_crt, load_private_key_file, sign_folders
from nvflare.private.fed.server.job_meta_validator import JobMetaValidator
from nvflare.private.fed.utils.fed_utils import require_signed_jobs
from nvflare.security.study_registry import StudyRegistryService

SERVICE_SUBMITTER = "scheduled-job-submitter"
_TICK_S = 5.0
_FINISHED_PREFIX = "FINISHED:"
SIGNER_DIR = Path("private") / "scheduled_jobs"


def _zip_with_folder_entries(base: str, job_name: str) -> bytes:
    """The job validator looks for folder entries, so write them explicitly."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for current, folders, names in os.walk(os.path.join(base, job_name)):
            folders.sort()
            zf.writestr(f"{os.path.relpath(current, base)}/", "")
            for name in sorted(names):
                zf.write(os.path.join(current, name), os.path.relpath(os.path.join(current, name), base))
    return buffer.getvalue()


class PeriodicJobSubmitter(FLComponent):
    def __init__(
        self,
        recipe: str,
        recipe_args: dict | None = None,
        interval_s: int = 86400,
        studies: list[str] | None = None,
        jitter_fraction: float = 0.1,
        startup_delay_s: float = 60.0,
        submit_timeout_s: int = 3600,
        signer_key: str | None = None,
        signer_cert: str | None = None,
    ):
        """Submits `recipe(**recipe_args)` to each Study every interval_s.

        Args:
            recipe: the dotted path of a Recipe class.
            studies: the Studies to submit in. None means every registered Study (the default Study when there is
                no registry).
            startup_delay_s: the first job of each Study starts within this many seconds of the server starting.
            submit_timeout_s: a job that has not started by then is aborted, so the next one can be submitted.
            signer_key, signer_cert: paths relative to the server workspace of the key and certificate of an
                admin-role service identity that signs the jobs. Both must be under private/scheduled_jobs/,
                which is never mounted or bundled into a job.
        """
        super().__init__()
        self._recipe = recipe
        self._recipe_args = dict(recipe_args or {})
        self._interval_s = interval_s
        self._studies = None if studies is None else list(studies)
        self._jitter_fraction = jitter_fraction
        self._startup_delay_s = startup_delay_s
        self._submit_timeout_s = submit_timeout_s
        self._signer_paths = (signer_key, signer_cert)
        self._signer: tuple[str, str] | None = None
        self._submitter = (SERVICE_SUBMITTER, "", "project_admin")  # (name, org, role) recorded on each job
        self._engine = None
        self._stop = threading.Event()
        self._running: dict[str, str] = {}  # study -> id of its job that has not finished
        self._next: dict[str, datetime] = {}  # study -> when its next job is due
        self._now = lambda: datetime.now(timezone.utc)

    @property
    def studies(self) -> list[str]:
        registry = StudyRegistryService.get_registry()
        return (
            list(self._studies)
            if self._studies is not None
            else sorted(registry.get_studies()) if registry else [DEFAULT_STUDY]
        )

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.SYSTEM_START:
            self._engine = fl_ctx.get_engine()
            self._load_signer(fl_ctx.get_workspace())
            threading.Thread(target=self._run, name="periodic-job-submitter", daemon=True).start()
        elif event_type == EventType.SYSTEM_END:
            self._stop.set()

    def _load_signer(self, workspace):
        self._signer = None
        key, cert = self._signer_paths
        if (key is None) != (cert is None):
            raise ValueError("set both signer_key and signer_cert")
        if key is None or cert is None:
            if require_signed_jobs(workspace):
                self.logger.error("scheduled jobs are unsigned but this server requires signed jobs: set signer_key")
            return
        root = Path(workspace.get_root_dir()).resolve()
        private_root = root / SIGNER_DIR
        paths = []
        for value in (key, cert):
            path = Path(os.path.abspath(root / value))
            resolved = path.resolve()
            if not path.is_relative_to(private_root) or not resolved.is_relative_to(private_root):
                raise ValueError("signer files must be under the server workspace's private/scheduled_jobs directory")
            paths.append(str(resolved))
        key, cert = paths
        info = get_identity_info(cert_to_dict(load_crt(cert)))
        self._signer = (key, cert)
        self._submitter = (info[IdentityKey.NAME], info[IdentityKey.ORG] or "", info[IdentityKey.ROLE])

    def _run(self):
        while not self._stop.wait(_TICK_S):
            try:
                self._tick()
            except Exception as e:
                self.logger.warning(f"scheduled job submission failed: {type(e).__name__}")

    def _tick(self):
        now = self._now()
        self._reap()
        for study in self.studies:
            if study not in self._next:
                self._next[study] = now + timedelta(seconds=random.uniform(0, self._startup_delay_s))
            if study in self._running or now < self._next[study]:
                continue
            # The next one is due an interval from now whatever happens, so a failing submit retries then.
            self._next[study] = now + timedelta(
                seconds=self._interval_s * random.uniform(1 - self._jitter_fraction, 1 + self._jitter_fraction)
            )
            try:
                self._running[study] = self._submit(study)
            except Exception as e:
                self.logger.warning(f"scheduled job could not be submitted: {type(e).__name__}")

    def _reap(self):
        """Forget a job that finished or is gone, and abort one that never started."""
        for study, job_id in list(self._running.items()):
            with self._engine.new_context() as fl_ctx:
                stored = self._job_manager().get_job(job_id, fl_ctx)
                status = (stored.meta.get(JobMetaKey.STATUS.value) or "") if stored is not None else None
                if status is None or status.startswith(_FINISHED_PREFIX):
                    del self._running[study]
                elif status == RunStatus.SUBMITTED and (
                    self._now().timestamp() - stored.meta.get(JobMetaKey.SUBMIT_TIME.value, 0.0)
                    > self._submit_timeout_s
                ):
                    # Through the job runner, which knows whether it is starting the job right now; deleting the
                    # job under it would break the start.
                    runner = self._engine.job_runner
                    if runner.abort_before_start(job_id, fl_ctx) is not runner.STARTING:
                        del self._running[study]

    def _job_manager(self):
        return self._engine.get_component(SystemComponents.JOB_MANAGER)

    def _build_zip(self) -> tuple[str, bytes]:
        recipe = load_class(self._recipe)(**self._recipe_args)
        with tempfile.TemporaryDirectory() as tmp:
            recipe.export(tmp)
            (job_name,) = os.listdir(tmp)  # the export writes one folder, named after the job
            if self._signer is not None:
                key_file, cert_file = self._signer
                sign_folders(os.path.join(tmp, job_name), load_private_key_file(key_file), cert_file)
            return job_name, _zip_with_folder_entries(tmp, job_name)

    def _submit(self, study: str) -> str:
        """Validate the job, let the components that guard every submission (SUBMIT_JOB) block it, then store it."""
        job_name, zip_bytes = self._build_zip()
        valid, error, meta = JobMetaValidator().validate(job_name, zip_bytes)
        if not valid:
            raise ValueError(f"scheduled job is invalid: {error}")
        meta.update(
            {
                JobMetaKey.STUDY.value: study,
                JobMetaKey.SUBMITTER_NAME.value: self._submitter[0],
                JobMetaKey.SUBMITTER_ORG.value: self._submitter[1],
                JobMetaKey.SUBMITTER_ROLE.value: self._submitter[2],
                JobMetaKey.JOB_FOLDER_NAME.value: job_name,
            }
        )
        with self._engine.new_context() as fl_ctx:
            fl_ctx.set_prop(FLContextKey.JOB_META, meta, private=True, sticky=False)
            self._engine.fire_event(EventType.SUBMIT_JOB, fl_ctx)
            block_reason = fl_ctx.get_prop(FLContextKey.JOB_BLOCK_REASON)
            if block_reason:
                raise ValueError(f"scheduled job is blocked: {block_reason}")
            return self._job_manager().create(meta, zip_bytes, fl_ctx)[JobMetaKey.JOB_ID.value]

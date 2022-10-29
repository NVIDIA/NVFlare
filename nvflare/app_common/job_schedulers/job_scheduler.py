# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import threading
import time

from typing import Dict, List, Optional, Tuple

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import ALL_SITES, Job, JobMetaKey
from nvflare.apis.job_scheduler_spec import DispatchInfo, JobSchedulerSpec
from nvflare.apis.server_engine_spec import ServerEngineSpec

SERVER_SITE_NAME = "server"


class DefaultJobScheduler(JobSchedulerSpec, FLComponent):
    def __init__(
        self,
        max_jobs: int=1,
        max_schedule_count: int=10,
        min_schedule_interval: float=10.0,
        max_schedule_interval: float = 600.0,
    ):
        """
        Create a DefaultJobScheduler
        Args:
            max_jobs: max number of concurrent jobs allowed
            max_schedule_count: max number of times to try to schedule a job
            min_schedule_interval: min interval between two schedules
            max_schedule_interval: max interval between two schedules
        """
        super().__init__()
        self.max_jobs = max_jobs
        self.max_schedule_count = max_schedule_count
        self.min_schedule_interval = min_schedule_interval
        self.max_schedule_interval = max_schedule_interval
        self.scheduled_jobs = []
        self.lock = threading.Lock()

    def _check_client_resources(
            self,
            job_id: str,
            resource_reqs: Dict[str, dict],
            fl_ctx: FLContext) -> Dict[str, Tuple[bool, str]]:
        """Checks resources on each site.

        Args:
            resource_reqs (dict): {client_name: resource_requirements}

        Returns:
            A dict of {client_name: client_check_result}.
                client_check_result is a tuple of (is_resource_enough, token);
                is_resource_enough is a bool indicates whether there is enough resources;
                token is for resource reservation / cancellation for this check request.
        """
        engine = fl_ctx.get_engine()
        if not isinstance(engine, ServerEngineSpec):
            raise RuntimeError(f"engine inside fl_ctx should be of type ServerEngineSpec, but got {type(engine)}.")

        result = engine.check_client_resources(job_id, resource_reqs)
        self.log_debug(fl_ctx, f"check client resources result: {result}")

        return result

    def _cancel_resources(
        self, resource_reqs: Dict[str, dict], resource_check_results: Dict[str, Tuple[bool, str]], fl_ctx: FLContext
    ):
        """Cancels any reserved resources based on resource check results.

        Args:
            resource_reqs (dict): {client_name: resource_requirements}
            resource_check_results: A dict of {client_name: client_check_result}
                where client_check_result is a tuple of {is_resource_enough, resource reserve token if any}
            fl_ctx: FL context
        """
        engine = fl_ctx.get_engine()
        if not isinstance(engine, ServerEngineSpec):
            raise RuntimeError(f"engine inside fl_ctx should be of type ServerEngineSpec, but got {type(engine)}.")

        engine.cancel_client_resources(resource_check_results, resource_reqs)
        self.log_debug(fl_ctx, f"cancel client resources using check results: {resource_check_results}")
        return False, None

    def _try_job(self, job: Job, fl_ctx: FLContext) -> (bool, Optional[Dict[str, DispatchInfo]]):
        engine = fl_ctx.get_engine()
        online_clients = engine.get_clients()
        online_site_names = [x.name for x in online_clients]

        if not job.deploy_map:
            raise RuntimeError(f"Job '{job.job_id}' does not have deploy_map, can't be scheduled.")

        applicable_sites = []
        sites_to_app = {}
        for app_name in job.deploy_map:
            for site_name in job.deploy_map[app_name]:
                if site_name.upper() == ALL_SITES:
                    # deploy_map: {"app_name": ["ALL_SITES"]} will be treated as deploying to all online clients
                    applicable_sites = online_site_names
                    sites_to_app = {x: app_name for x in online_site_names}
                    sites_to_app[SERVER_SITE_NAME] = app_name
                elif site_name in online_site_names:
                    applicable_sites.append(site_name)
                    sites_to_app[site_name] = app_name
                elif site_name == SERVER_SITE_NAME:
                    sites_to_app[SERVER_SITE_NAME] = app_name
        self.log_debug(fl_ctx, f"Job {job.job_id} is checking against applicable sites: {applicable_sites}")

        required_sites = job.required_sites if job.required_sites else []
        if required_sites:
            for s in required_sites:
                if s not in applicable_sites:
                    self.log_debug(fl_ctx, f"Job {job.job_id} can't be scheduled: required site {s} is not connected.")
                    return False, None

        if job.min_sites and len(applicable_sites) < job.min_sites:
            self.log_debug(
                fl_ctx,
                f"Job {job.job_id} can't be scheduled: connected sites ({len(applicable_sites)}) "
                f"are less than min_sites ({job.min_sites}).",
            )
            return False, None

        # we are assuming server resource is sufficient
        resource_reqs = {}
        for site_name in applicable_sites:
            if site_name in job.resource_spec:
                resource_reqs[site_name] = job.resource_spec[site_name]
            else:
                resource_reqs[site_name] = {}

        job_participants = [fl_ctx.get_identity_name(default="server")]
        job_participants.extend(applicable_sites)

        fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, job.job_id, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.CLIENT_RESOURCE_SPECS, resource_reqs, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.JOB_PARTICIPANTS, job_participants, private=True, sticky=False)
        self.fire_event(EventType.BEFORE_CHECK_CLIENT_RESOURCES, fl_ctx)

        block_reason = fl_ctx.get_prop(FLContextKey.JOB_BLOCK_REASON)
        if block_reason:
            # cannot schedule this job
            self.log_info(fl_ctx, f"Job {job.job_id} can't be scheduled: {block_reason}")
            return False, None

        resource_check_results = self._check_client_resources(
            job_id=job.job_id,
            resource_reqs=resource_reqs,
            fl_ctx=fl_ctx)

        if not resource_check_results:
            self.log_debug(fl_ctx, f"Job {job.job_id} can't be scheduled: resource check results is None or empty.")
            return False, None

        required_sites_not_enough_resource = list(required_sites)
        num_sites_ok = 0
        sites_dispatch_info = {}
        for site_name, check_result in resource_check_results.items():
            is_resource_enough, token = check_result
            if is_resource_enough:
                sites_dispatch_info[site_name] = DispatchInfo(
                    app_name=sites_to_app[site_name],
                    resource_requirements=resource_reqs[site_name],
                    token=token,
                )
                num_sites_ok += 1
                if site_name in required_sites:
                    required_sites_not_enough_resource.remove(site_name)

        if num_sites_ok < job.min_sites:
            self.log_debug(fl_ctx, f"Job {job.job_id} can't be scheduled: not enough sites have enough resources.")
            return self._cancel_resources(
                resource_reqs=job.resource_spec, resource_check_results=resource_check_results, fl_ctx=fl_ctx
            )

        if required_sites_not_enough_resource:
            self.log_debug(
                fl_ctx,
                f"Job {job.job_id} can't be scheduled: required sites: {required_sites_not_enough_resource}"
                f" don't have enough resources.",
            )
            return self._cancel_resources(
                resource_reqs=job.resource_spec, resource_check_results=resource_check_results, fl_ctx=fl_ctx
            )

        # add server dispatch info
        sites_dispatch_info[SERVER_SITE_NAME] = DispatchInfo(
            app_name=sites_to_app[SERVER_SITE_NAME], resource_requirements={}, token=None
        )

        return True, sites_dispatch_info

    def _exceed_max_jobs(self, fl_ctx: FLContext) -> bool:
        exceed_limit = False
        with self.lock:
            if len(self.scheduled_jobs) >= self.max_jobs:
                self.log_debug(
                    fl_ctx,
                    f"Skipping schedule job because scheduled_jobs ({len(self.scheduled_jobs)}) "
                    f"is greater than max_jobs ({self.max_jobs})",
                )
                exceed_limit = True
        return exceed_limit

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.JOB_STARTED:
            with self.lock:
                job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
                if job_id not in self.scheduled_jobs:
                    self.scheduled_jobs.append(job_id)
        elif event_type == EventType.JOB_COMPLETED or event_type == EventType.JOB_ABORTED:
            with self.lock:
                job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
                if job_id in self.scheduled_jobs:
                    self.scheduled_jobs.remove(job_id)

    def schedule_job(
        self, job_candidates: List[Job], fl_ctx: FLContext
    ) -> (Optional[Job], Optional[Dict[str, DispatchInfo]], Optional[List[Job]], Optional[List[Job]]):
        self.log_debug(fl_ctx, f"Current scheduled_jobs is {self.scheduled_jobs}")
        if self._exceed_max_jobs(fl_ctx=fl_ctx):
            self.log_debug(
                fl_ctx,
                f"skipped scheduling since there are {self.max_jobs} concurrent job(s) already")
            return None, None, None, None

        # sort by submitted time
        job_candidates.sort(key=lambda j: j.meta.get(JobMetaKey.SUBMIT_TIME.value, 0.0))
        engine = fl_ctx.get_engine()
        failed_jobs = []
        blocked_jobs = []

        for job in job_candidates:
            schedule_count = job.meta.get(JobMetaKey.SCHEDULE_COUNT.value, 0)
            if schedule_count >= self.max_schedule_count:
                self.log_info(
                    fl_ctx,
                    f"skipped job {job.job_id} since it exceeded max schedule count {self.max_schedule_count}")
                blocked_jobs.append(job)
                continue

            if schedule_count < 0:
                schedule_count = 0

            last_schedule_time = job.meta.get(JobMetaKey.LAST_SCHEDULE_TIME.value, 0.0)
            time_since_last_schedule = time.time() - last_schedule_time
            n = 0 if schedule_count == 0 else schedule_count - 1
            required_interval = min(self.max_schedule_interval, (2 ** n) * self.min_schedule_interval)
            if time_since_last_schedule < required_interval:
                # do not schedule again too soon
                continue

            with engine.new_context() as ctx:
                ok, sites_dispatch_info = self._try_job(job, ctx)
                self.log_debug(ctx, f"Try to schedule job {job.job_id}, get result: {ok}, {sites_dispatch_info}.")
                if ok:
                    return job, sites_dispatch_info, failed_jobs, blocked_jobs
                else:
                    failed_jobs.append(job)

        self.log_debug(fl_ctx, "No job is scheduled.")
        return None, None, failed_jobs, blocked_jobs

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

import pickle
import threading
from typing import Dict, List, Optional, Tuple

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import Job
from nvflare.apis.job_scheduler_spec import DispatchInfo, JobSchedulerSpec
from nvflare.private.fed.server.admin import ClientReply
from nvflare.private.fed.server.server_engine import ServerEngine
from nvflare.private.scheduler_constants import ShareableHeader


def _check_client_resources(resource_reqs: Dict[str, dict], fl_ctx: FLContext) -> Dict[str, Tuple[bool, str]]:
    """Checks resources on each site.

    Args:
        resource_reqs (dict): {client_name: resource_requirements}

    Returns:
        A dict of {client_name: client_check_result}
        where client_check_result is a tuple of {client check OK, resource reserve token if any}
    """
    engine = fl_ctx.get_engine()
    if not isinstance(engine, ServerEngine):
        raise RuntimeError(f"engine inside fl_ctx should be of type ServerEngine, but got {type(engine)}.")

    replies: List[ClientReply] = engine.check_client_resources(resource_reqs)

    result = {}
    for r in replies:
        site_name = engine.get_client_name_from_token(r.client_token)
        if r.reply:
            resp = pickle.loads(r.reply.body)
            result[site_name] = (
                resp.get_header(ShareableHeader.CHECK_RESOURCE_RESULT, False),
                resp.get_header(ShareableHeader.RESOURCE_RESERVE_TOKEN, ""),
            )
        else:
            result[site_name] = (False, "")

    return result


def _cancel_resources(
    resource_reqs: Dict[str, dict], resource_check_results: Dict[str, Tuple[bool, str]], fl_ctx: FLContext
):
    """Cancels any reserved resources based on resource check results.

    Args:
        resource_reqs (dict): {client_name: resource_requirements}
        resource_check_results: A dict of {client_name: client_check_result}
            where client_check_result is a tuple of {client check OK, resource reserve token if any}
        fl_ctx: FL context
    """
    engine = fl_ctx.get_engine()
    if not isinstance(engine, ServerEngine):
        raise RuntimeError(f"engine inside fl_ctx should be of type ServerEngine, but got {type(engine)}.")

    engine.cancel_client_resources(resource_check_results, resource_reqs)
    return False, None


def _try_job(job: Job, fl_ctx) -> (bool, Optional[Dict[str, DispatchInfo]]):
    if not job.resource_spec:
        all_sites = set()
        for i in job.deploy_map:
            all_sites.update(job.deploy_map[i])
        all_sites.update(job.required_sites)
        return True, {x: DispatchInfo(resource_requirements={}, token="") for x in all_sites}

    # we are assuming server resource is sufficient
    resource_check_results = _check_client_resources(resource_reqs=job.resource_spec, fl_ctx=fl_ctx)

    if not resource_check_results:
        return False, None

    if len(resource_check_results) < job.min_sites:
        return _cancel_resources(
            resource_reqs=job.resource_spec, resource_check_results=resource_check_results, fl_ctx=fl_ctx
        )

    required_sites_received = 0
    num_sites_ok = 0
    sites_dispatch_info = {}
    for site_name, check_result in resource_check_results.items():
        if check_result[0]:
            sites_dispatch_info[site_name] = DispatchInfo(
                resource_requirements=job.resource_spec[site_name], token=check_result[1]
            )
            num_sites_ok += 1
            if site_name in job.required_sites:
                required_sites_received += 1

    if num_sites_ok < job.min_sites:
        return _cancel_resources(
            resource_reqs=job.resource_spec, resource_check_results=resource_check_results, fl_ctx=fl_ctx
        )

    if required_sites_received < len(job.required_sites):
        return _cancel_resources(
            resource_reqs=job.resource_spec, resource_check_results=resource_check_results, fl_ctx=fl_ctx
        )

    return True, sites_dispatch_info


class DefaultJobScheduler(JobSchedulerSpec, FLComponent):
    def __init__(
        self,
        client_req_timeout: float = 1.0,
        max_jobs: int = 10,
    ):
        super().__init__()
        self.client_req_timeout = client_req_timeout
        self.max_jobs = max_jobs
        self.scheduled_jobs = []
        self.lock = threading.Lock()

    def schedule_job(
        self, job_candidates: List[Job], fl_ctx: FLContext
    ) -> (Optional[Job], Optional[Dict[str, DispatchInfo]]):
        if len(self.scheduled_jobs) >= self.max_jobs:
            return None, None

        for job in job_candidates:
            ok, sites_dispatch_info = _try_job(job, fl_ctx)
            if ok:
                with self.lock:
                    self.scheduled_jobs.append(job)
                return job, sites_dispatch_info
        return None, None

    def remove_job(self, job: Job):
        with self.lock:
            if job in self.scheduled_jobs:
                self.scheduled_jobs.remove(job)

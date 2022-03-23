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

import copy
from enum import Enum
from typing import Dict, List

from .run_manager_spec import Resources


class RunStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    APPROVED = "APPROVED"
    DISPATCHED = "DISPATCHED"
    RUNNING = "RUNNING"
    FINISHED_COMPLETED = "FINISHED:COMPLETED"
    FINISHED_ABORTED = "FINISHED:ABORTED"


class JobMetaKey(str, Enum):
    STUDY_NAME = "study_name"
    JOB_ID = "job_id"
    JOB_NAME = "job_name"
    STATUS = "status"
    DEPLOY_MAP = "deploy_map"
    RESOURCE_SPEC = "resource_spec"
    CONTENT_LOCATION = "content_location"
    RESULT_LOCATION = "result_location"
    APPROVALS = "approvals"
    MIN_CLIENTS = "min_clients"
    MANDATORY_CLIENTS = "mandatory_clients"
    SUBMIT_TIME = "submit_time"
    SUBMIT_TIME_ISO = "submit_time_iso"

    def __repr__(self):
        return self.value


class Job:
    def __init__(self, job_id, study_name, resource_spec, deploy_map, meta):
        self.job_id = job_id
        self.study = study_name
        # self.num_clients = num_clients  # some way to specify minimum clients needed sites
        self.resource_spec = resource_spec  # resource_requirements should be {client name: resource}
        self.deploy_map = deploy_map  # should be {app name: a list of sites}

        self.meta = meta
        self.dispatcher_id = None
        self.dispatch_time = None

        self.submit_time = None

        self.run_record = None  # run number, dispatched time/UUID, finished time, completion code (normal, aborted)

    def get_num_clients(self):
        return self.num_clients

    def get_deployment(self) -> Dict[str, List[str]]:
        """Return deployment configuration."""
        return self.deploy_map

    def get_resource_requirements(self) -> Dict[str, Resources]:
        """Return app resource requirements."""
        return self.resource_spec

    def __eq__(self, other):
        return self.job_id == other.job_id


def job_from_meta(meta: dict) -> Job:
    """Convert information in meta into a Job object.

    Args:
        meta: dict of meta information

    Returns:
        Job object.

    """
    job = Job(
        meta.get(JobMetaKey.JOB_ID),
        meta.get(JobMetaKey.STUDY_NAME),
        meta.get(JobMetaKey.RESOURCE_SPEC),
        meta.get(JobMetaKey.DEPLOY_MAP),
        meta,
    )
    return job


def get_site_require_resource_from_job(job: Job) -> Dict[str, Resources]:
    """Get the total resource needed by each site to run this Job."""
    required_resources = job.get_resource_requirements()
    deployment = job.get_deployment()

    total_required_resources = {}  # {site name: total resources}
    for app in required_resources:
        for site_name in deployment[app]:
            if site_name not in total_required_resources:
                total_required_resources[site_name] = copy.deepcopy(required_resources[app])
            else:
                total_required_resources[site_name] = total_required_resources[site_name] + required_resources[app]
    return total_required_resources

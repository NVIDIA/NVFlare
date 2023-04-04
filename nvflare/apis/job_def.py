# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from enum import Enum
from typing import Dict, List, Optional

from nvflare.apis.fl_constant import SystemComponents
from nvflare.apis.fl_context import FLContext

# this is treated as all online sites in job deploy_map
ALL_SITES = "@ALL"
SERVER_SITE_NAME = "server"


class RunStatus(str, Enum):

    SUBMITTED = "SUBMITTED"
    APPROVED = "APPROVED"
    DISPATCHED = "DISPATCHED"
    RUNNING = "RUNNING"
    FINISHED_COMPLETED = "FINISHED:COMPLETED"
    FINISHED_ABORTED = "FINISHED:ABORTED"
    FINISHED_EXECUTION_EXCEPTION = "FINISHED:EXECUTION_EXCEPTION"
    FINISHED_ABNORMAL = "FINISHED:ABNORMAL"
    FINISHED_CANT_SCHEDULE = "FINISHED:CAN_NOT_SCHEDULE"
    FAILED_TO_RUN = "FINISHED:FAILED_TO_RUN"
    ABANDONED = "FINISHED:ABANDONED"


class JobDataKey(str, Enum):
    DATA = "data"
    META = "meta"
    JOB_DATA = "job_data_"
    WORKSPACE_DATA = "workspace_data_"


class JobMetaKey(str, Enum):
    JOB_ID = "job_id"
    JOB_NAME = "name"
    JOB_FOLDER_NAME = "job_folder_name"
    SUBMITTER_NAME = "submitter_name"
    SUBMITTER_ORG = "submitter_org"
    SUBMITTER_ROLE = "submitter_role"
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
    START_TIME = "start_time"
    DURATION = "duration"
    JOB_DEPLOY_DETAIL = "job_deploy_detail"
    SCHEDULE_COUNT = "schedule_count"
    SCOPE = "scope"
    CLONED_FROM = "cloned_from"
    LAST_SCHEDULE_TIME = "last_schedule_time"
    SCHEDULE_HISTORY = "schedule_history"

    def __repr__(self):
        return self.value


class TopDir(object):
    JOB = "job"
    WORKSPACE = "workspace"


class Job:
    def __init__(
        self,
        job_id: str,
        resource_spec: Dict[str, Dict],
        deploy_map: Dict[str, List[str]],
        meta,
        min_sites: int = 1,
        required_sites: Optional[List[str]] = None,
    ):
        """Job object containing the job metadata.

        Args:
            job_id: Job ID
            resource_spec: Resource specification with information on the resources of each client
            deploy_map: Deploy map specifying each app and the sites that it should be deployed to
            meta: full contents of the persisted metadata for the job for persistent storage
            min_sites (int): minimum number of sites
            required_sites: A list of required site names
        """
        self.job_id = job_id
        self.resource_spec = resource_spec  # resource_requirements should be {site name: resource}
        self.deploy_map = deploy_map  # should be {app name: a list of sites}

        self.meta = meta
        self.min_sites = min_sites
        self.required_sites = required_sites
        if not self.required_sites:
            self.required_sites = []

        self.dispatcher_id = None
        self.dispatch_time = None

        self.submit_time = None

        self.run_record = None  # job id, dispatched time/UUID, finished time, completion code (normal, aborted)
        self.run_aborted = False

    def get_deployment(self) -> Dict[str, List[str]]:
        """Returns the deployment configuration.

        ::

            "deploy_map": {
                "hello-numpy-sag-server": [
                  "server"
                ],
                "hello-numpy-sag-client": [
                  "client1",
                  "client2"
                ],
                "hello-numpy-sag-client3": [
                  "client3"
                ]
              },

        Returns:
            Contents of deploy_map as a dictionary of strings of app names with their corresponding sites
        """
        return self.deploy_map

    def get_application(self, app_name, fl_ctx: FLContext) -> bytes:
        """Get the application content in bytes for the specified participant."""
        # application_name = self.get_application_name(participant)
        engine = fl_ctx.get_engine()
        job_def_manager = engine.get_component(SystemComponents.JOB_MANAGER)
        # # if not isinstance(job_def_manager, JobDefManagerSpec):
        # #     raise TypeError(f"job_def_manager must be JobDefManagerSpec type. Got: {type(job_def_manager)}")
        return job_def_manager.get_app(self, app_name, fl_ctx)

    def get_application_name(self, participant):
        """Get the application name for the specified participant."""
        for app in self.deploy_map:
            for site in self.deploy_map[app]:
                if site == participant:
                    return app
        return None

    def get_resource_requirements(self):
        """Returns app resource requirements.

        Returns:
            A dict of {site_name: resource}
        """
        return self.resource_spec

    def __eq__(self, other):
        return self.job_id == other.job_id


def job_from_meta(meta: dict) -> Job:
    """Converts information in meta into a Job object.

    Args:
        meta: dict of meta information

    Returns:
        A Job object.
    """
    job = Job(
        job_id=meta.get(JobMetaKey.JOB_ID, ""),
        resource_spec=meta.get(JobMetaKey.RESOURCE_SPEC, {}),
        deploy_map=meta.get(JobMetaKey.DEPLOY_MAP, {}),
        meta=meta,
        min_sites=meta.get(JobMetaKey.MIN_CLIENTS, 1),
        required_sites=meta.get(JobMetaKey.MANDATORY_CLIENTS, []),
    )
    return job

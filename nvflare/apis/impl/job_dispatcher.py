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

import logging
import os.path
import shutil
import uuid
from typing import Dict, List

from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.scheduler_constants import AuxChannelTopic
from nvflare.apis.impl.scheduler_constants import FLContextKey as SchedulerFLContextKey
from nvflare.apis.impl.scheduler_constants import ShareableHeader
from nvflare.apis.job_def import Job
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec
from nvflare.apis.job_dispatcher_spec import DispatchStatus, JobDispatcherSpec
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import Shareable
from nvflare.fuel.hci.zip_utils import unzip_all_from_bytes

SERVER_NAME = "server"


def get_site_to_app_map(deployment: Dict[str, List[str]]):
    result = {}
    for app_name, site_names in deployment.items():
        for site_name in site_names:
            if site_name in result:
                raise RuntimeError("Each site can only run one app in a job.")
            else:
                result[site_name] = app_name
    return result


def dispatch_to_server(app_name: str, app_bytes: bytes, fl_ctx: FLContext) -> DispatchStatus:
    engine = fl_ctx.get_engine()
    if not isinstance(engine, ServerEngineSpec):
        raise RuntimeError(f"engine inside fl_ctx should be of type ServerEngineSpec, but got {type(engine)}.")
    try:
        run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        run_dir = engine.get_workspace().get_run_dir(run_number)
        app_folder = os.path.join(run_dir, app_name)
        if os.path.exists(app_folder):
            shutil.rmtree(app_folder)
        unzip_all_from_bytes(app_bytes, app_folder)
    except:
        return DispatchStatus.DEPLOY_FAILED

    try:
        # TODO:: do we start app here????
        #  note that start_app_on_server is defined in ServerEngineInternalSpec instead of ServerEngineSpec...
        print("start server app")
    except:
        return DispatchStatus.START_FAILED

    return DispatchStatus.SUCCESS


def _send_req_to_sites(
    request: Shareable, topic: str, sites: List[str], fl_ctx: FLContext, timeout
) -> Dict[str, Shareable]:
    engine = fl_ctx.get_engine()
    if not isinstance(engine, ServerEngineSpec):
        raise RuntimeError(f"engine inside fl_ctx should be of type ServerEngineSpec, but got {type(engine)}.")
    # result is {client_name: Shareable} of each site's result
    result = engine.send_aux_request(targets=sites, topic=topic, request=request, timeout=timeout, fl_ctx=fl_ctx)
    return result


def dispatch_to_client(app_name: str, app_bytes: bytes, site: str, timeout, fl_ctx: FLContext) -> DispatchStatus:
    request = Shareable()
    request.set_header(ShareableHeader.APP_NAME, app_name)
    request.set_header(ShareableHeader.APP_BYTES, app_bytes)
    reply = _send_req_to_sites(
        request=request, sites=[site], topic=AuxChannelTopic.DISPATCH_APP, timeout=timeout, fl_ctx=fl_ctx
    )
    if reply[site].get_return_code() != ReturnCode.OK:
        return DispatchStatus.DEPLOY_FAILED
    result = reply[site].get_header(ShareableHeader.DISPATCH_STATUS)
    return result


def stop_server_app(app_name: str, fl_ctx: FLContext) -> bool:
    engine = fl_ctx.get_engine()
    if not isinstance(engine, ServerEngineSpec):
        raise RuntimeError(f"engine inside fl_ctx should be of type ServerEngineSpec, but got {type(engine)}.")
    # TODO:: abort app on server is defined in ServerEngineInternalSpec, what should we do?
    #   we should be able to pass app_name in....
    #   engine.abort_app_on_server()
    return True


def stop_client_app(app_name: str, site: str, timeout, fl_ctx: FLContext) -> bool:
    request = Shareable()
    request.set_header(ShareableHeader.APP_NAME, app_name)
    reply = _send_req_to_sites(
        request=request, sites=[site], topic=AuxChannelTopic.STOP_APP, timeout=timeout, fl_ctx=fl_ctx
    )
    if reply[site].get_return_code() != ReturnCode.OK:
        return False
    return True


class JobDispatcher(JobDispatcherSpec):
    def __init__(self, client_req_timeout: int = 60):
        self.id = uuid.uuid4()
        self.client_req_timeout = client_req_timeout
        self.logger = logging.getLogger(str(self.__class__))

    def dispatch_app(self, app_name: str, app_bytes: bytes, site: str, fl_ctx: FLContext) -> DispatchStatus:
        if site == SERVER_NAME:
            # dispatch app to server
            return dispatch_to_server(app_name=app_name, app_bytes=app_bytes, fl_ctx=fl_ctx)
        else:
            # dispatch app to clients
            return dispatch_to_client(
                app_name=app_name, app_bytes=app_bytes, site=site, timeout=self.client_req_timeout, fl_ctx=fl_ctx
            )

    def dispatch(self, job: Job, sites: List[str], fl_ctx: FLContext) -> Dict[str, DispatchStatus]:
        result = {}
        deployment = job.get_deployment()
        site_to_app = get_site_to_app_map(deployment)
        job_manager = fl_ctx.get_prop(SchedulerFLContextKey.JOB_MANAGER)
        if not isinstance(job_manager, JobDefManagerSpec):
            raise RuntimeError(f"job_manager should be of type JobDefManagerSpec, but got {type(job_manager)}.")
        app_to_bytes = job_manager.get_apps(job)
        for site_name in sites:
            if site_name not in deployment:
                raise RuntimeError("Site ({}) is not in job deployment.".format(site_name))
            app_name = site_to_app[site_name]
            result[site_name] = self.dispatch_app(
                app_name=app_name, app_bytes=app_to_bytes[app_name], site=site_name, fl_ctx=fl_ctx
            )
        return result

    def stop_app(self, app_name: str, site: str, fl_ctx: FLContext) -> bool:
        if site == SERVER_NAME:
            return stop_server_app(app_name=app_name, fl_ctx=fl_ctx)
        else:
            return stop_client_app(app_name=app_name, site=site, timeout=self.client_req_timeout, fl_ctx=fl_ctx)

    def stop(self, job: Job, sites: List[str], fl_ctx: FLContext) -> Dict[str, bool]:
        result = {}
        deployment = job.get_deployment()
        site_to_app = get_site_to_app_map(deployment)
        for site_name in sites:
            if site_name not in deployment:
                raise RuntimeError("Site ({}) is not in job deployment.".format(site_name))
            app_name = site_to_app[site_name]
            result[site_name] = self.stop_app(app_name=app_name, site=site_name, fl_ctx=fl_ctx)
        return result

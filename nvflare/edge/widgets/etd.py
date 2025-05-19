# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from random import randrange

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.edge.constants import EdgeApiStatus, EdgeContextKey, EdgeEventType, EdgeProtoKey
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.widgets.widget import Widget


class EdgeTaskDispatcher(Widget):
    """Edge Task Dispatcher (ETD) is to be used to dispatch a received edge request to a running job (CJ).
    ETD must be installed on CP before the CP is started.

    Note: ETD does not interact with edge devices directly. It's another component's responsibility (e.g. web agent)
    to interact with edge devices with whatever protocol between them.

    ETD indirectly interacts with edge-device-interacting component (also installed on the CP) via Flare Events:
        EdgeEventType.EDGE_JOB_REQUEST_RECEIVED for receiving job requests;
        EdgeEventType.EDGE_REQUEST_RECEIVED for receiving task requests;

    """

    def __init__(self, request_timeout: float = 10.0):
        Widget.__init__(self)
        self.request_timeout = request_timeout
        self.edge_jobs = {}  # edge_method => list of job_ids
        self.job_metas = {}  # job_id => job_meta
        self.lock = threading.Lock()

        self.register_event_handler(
            EventType.AFTER_JOB_LAUNCH,
            self._handle_job_launched,
        )
        self.register_event_handler(
            [EventType.JOB_COMPLETED, EventType.JOB_CANCELLED, EventType.JOB_ABORTED],
            self._handle_job_done,
        )
        self.register_event_handler(
            EdgeEventType.EDGE_JOB_REQUEST_RECEIVED,
            self._handle_edge_job_request,
        )
        self.register_event_handler(
            EdgeEventType.EDGE_REQUEST_RECEIVED,
            self._handle_edge_request,
        )
        self.logger.info("EdgeTaskDispatcher created!")

    def _add_job(self, job_meta: dict):
        with self.lock:
            edge_method = job_meta.get(JobMetaKey.EDGE_METHOD)
            if not edge_method:
                # this is not an edge job
                return

            jobs = self.edge_jobs.get(edge_method)
            if not jobs:
                jobs = []
                self.edge_jobs[edge_method] = jobs

            job_id = job_meta.get(JobMetaKey.JOB_ID)
            jobs.append(job_id)
            self.job_metas[job_id] = job_meta

    def _remove_job(self, job_id: str):
        with self.lock:
            if job_id in self.job_metas:
                del self.job_metas[job_id]

            # Delete this job from all methods
            for edge_method, jobs in list(self.edge_jobs.items()):
                assert isinstance(jobs, list)
                if jobs and job_id in jobs:
                    jobs.remove(job_id)
                    if not jobs:
                        # no more jobs for this edge method
                        self.edge_jobs.pop(edge_method)

    def _match_job(self, caps: dict):
        methods = caps.get("methods")
        with self.lock:
            for edge_method, jobs in self.edge_jobs.items():
                if edge_method in methods:
                    # pick one randomly
                    i = randrange(len(jobs))
                    return jobs[i]

            # no job matched
            return None

    def _find_job(self, job_id: str):
        with self.lock:
            for jobs in self.edge_jobs.values():
                if job_id in jobs:
                    return True
            return False

    def _handle_job_launched(self, event_type: str, fl_ctx: FLContext):
        self.logger.info(f"handling event {event_type}")
        job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        if not job_meta:
            self.logger.error(f"missing {FLContextKey.JOB_META} from fl_ctx for event {event_type}")
        else:
            self._add_job(job_meta)

    def _handle_job_done(self, event_type: str, fl_ctx: FLContext):
        self.logger.info(f"handling event {event_type}")
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
        if not job_id:
            self.logger.error(f"missing {FLContextKey.CURRENT_JOB_ID} from fl_ctx for event {event_type}")
        else:
            self._remove_job(job_id)

    def _handle_edge_job_request(self, event_type: str, fl_ctx: FLContext):
        self.logger.debug(f"handling event {event_type}")
        edge_capabilities = fl_ctx.get_prop(EdgeContextKey.EDGE_CAPABILITIES)
        if not edge_capabilities:
            self.logger.error(f"missing {EdgeContextKey.EDGE_CAPABILITIES} from fl_ctx for event {event_type}")
            self._set_edge_reply(EdgeApiStatus.INVALID_REQUEST, None, fl_ctx)
            return

        # find job for the caps
        job_id = self._match_job(edge_capabilities)
        if job_id:
            status = EdgeApiStatus.OK
        else:
            status = EdgeApiStatus.NO_JOB
        self._set_edge_reply(status, job_id, fl_ctx)
        fl_ctx.set_prop(FLContextKey.JOB_META, self.job_metas.get(job_id), private=True, sticky=False)

    @staticmethod
    def _set_edge_reply(status, data, fl_ctx: FLContext):
        """Prepare the reply to the edge device.

        Args:
            status:
            data:
            fl_ctx:

        Returns: None

        The "data" is response to the edge device.
        If is either generated by CJ or by self._handle_edge_job_request.
        If generated by self._handle_edge_job_request, the data is simpy the job id.

        If generated by CJ, the data is a dict with two elements:
            status: the processing status of the request
            response: the response data to the request. It is one of predefined XXXResponse

        Note that the "data" could be None in case the communication to CJ failed.

        Here we wrap the "data" in yet another dict with two elements:
            status: the communication status to CJ. This status is not to be confused with the status in data.
            data: the data received from CJ.

        """

        fl_ctx.set_prop(
            key=EdgeContextKey.REPLY_TO_EDGE,
            value={EdgeProtoKey.STATUS: status, EdgeProtoKey.DATA: data},
            private=True,
            sticky=False,
        )

    def _handle_edge_request(self, event_type: str, fl_ctx: FLContext):
        # try to find the job
        job_id = fl_ctx.get_prop(EdgeContextKey.JOB_ID)
        if not job_id:
            self.logger.error(f"handling event {event_type}: missing {EdgeContextKey.JOB_ID} from fl_ctx")
            self._set_edge_reply(EdgeApiStatus.INVALID_REQUEST, None, fl_ctx)
            return

        if not self._find_job(job_id):
            self._set_edge_reply(EdgeApiStatus.NO_JOB, None, fl_ctx)
            return

        # send edge request data to CJ
        edge_req_data = fl_ctx.get_prop(EdgeContextKey.REQUEST_FROM_EDGE)
        self.logger.debug(f"Sending edge request to CJ {job_id}")
        engine = fl_ctx.get_engine()
        reply = engine.send_to_job(
            job_id=job_id,
            channel=CellChannel.EDGE_REQUEST,
            topic="request",
            msg=new_cell_message({}, edge_req_data),
            timeout=self.request_timeout,
            optional=True,
        )

        assert isinstance(reply, CellMessage)
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        reply_data = reply.payload
        self.logger.debug(f"got edge result from CJ: {rc=} {reply_data=}")
        self._set_edge_reply(rc, reply_data, fl_ctx)

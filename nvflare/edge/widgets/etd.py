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
import time
from random import randrange

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.edge.constants import EdgeApiStatus, EdgeContextKey, EdgeEventType, EdgeMsgTopic
from nvflare.edge.web.models.capabilities import Capabilities
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_response import SelectionResponse
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.fuel.f3.cellnet.cell import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.widgets.widget import Widget


class EdgeTaskDispatcher(Widget):
    """Edge Task Dispatcher (ETD) is to be used to dispatch a received edge request to a running job (CJ).
    ETD must be installed on CP (local/resources.json) before the CP is started.

    Note: ETD does not interact with edge devices directly. It's another component's responsibility (e.g. web agent)
    to interact with edge devices with whatever protocol between them.

    ETD indirectly interacts with edge-device-interacting component (also installed on the CP) via Flare Events:
        EdgeEventType.EDGE_JOB_REQUEST_RECEIVED for receiving job requests;
        EdgeEventType.EDGE_TASK_REQUEST_RECEIVED for receiving task requests;
        EdgeEventType.EDGE_SELECTION_REQUEST_RECEIVED for receiving selection requests;
        EdgeEventType.EDGE_RESULT_REPORT_RECEIVED for receiving result reports;

    """

    def __init__(self, request_timeout: float = 5.0):
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
            EdgeEventType.EDGE_TASK_REQUEST_RECEIVED,
            self._handle_edge_request,
            msg_topic=EdgeMsgTopic.TASK_REQUEST,
            bad_req_reply=TaskResponse(EdgeApiStatus.INVALID_REQUEST),
            no_job_reply=TaskResponse(EdgeApiStatus.DONE),
            comm_err_reply=TaskResponse(EdgeApiStatus.RETRY),
        )
        self.register_event_handler(
            EdgeEventType.EDGE_SELECTION_REQUEST_RECEIVED,
            self._handle_edge_request,
            msg_topic=EdgeMsgTopic.SELECTION_REQUEST,
            bad_req_reply=SelectionResponse(EdgeApiStatus.INVALID_REQUEST),
            no_job_reply=SelectionResponse(EdgeApiStatus.DONE),
            comm_err_reply=SelectionResponse(EdgeApiStatus.RETRY),
        )
        self.register_event_handler(
            EdgeEventType.EDGE_RESULT_REPORT_RECEIVED,
            self._handle_edge_request,
            msg_topic=EdgeMsgTopic.RESULT_REPORT,
            bad_req_reply=ResultResponse(EdgeApiStatus.INVALID_REQUEST),
            no_job_reply=ResultResponse(EdgeApiStatus.DONE),
            comm_err_reply=ResultResponse(EdgeApiStatus.RETRY),
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

    def _match_job(self, caps: Capabilities):
        methods = caps.methods
        with self.lock:
            for edge_method, jobs in self.edge_jobs.items():
                if edge_method in methods:
                    # pick one randomly
                    i = randrange(len(jobs))
                    self.logger.info(f"matched job {jobs[i]}")
                    return jobs[i], edge_method

            # no job matched
            return None, None

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
            self.logger.info(f"adding job: {job_meta=}")
            self._add_job(job_meta)

    def _handle_job_done(self, event_type: str, fl_ctx: FLContext):
        self.logger.info(f"handling event {event_type}")
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
        if not job_id:
            self.logger.error(f"missing {FLContextKey.CURRENT_JOB_ID} from fl_ctx for event {event_type}")
        else:
            self._remove_job(job_id)

    def _handle_edge_job_request(self, event_type: str, fl_ctx: FLContext):
        self.logger.info(f"handling event {event_type}")
        req = fl_ctx.get_prop(EdgeContextKey.REQUEST_FROM_EDGE)
        assert isinstance(req, JobRequest)
        edge_capabilities = req.capabilities
        if not edge_capabilities:
            self.logger.error(f"missing 'capabilities' from JobRequest for event {event_type}")
            self._set_edge_reply(reply=JobResponse(EdgeApiStatus.INVALID_REQUEST), fl_ctx=fl_ctx)
            return

        # find job for the caps
        self.logger.info(f"trying to match job with caps: {edge_capabilities}")
        job_id, edge_method = self._match_job(edge_capabilities)
        if job_id:
            reply = JobResponse(
                status=EdgeApiStatus.OK,
                job_id=job_id,
                job_name=self.job_metas.get(job_id).get(JobMetaKey.JOB_NAME),
                job_data=self.job_metas.get(job_id),
                method=edge_method,
            )
        else:
            reply = JobResponse(EdgeApiStatus.NO_JOB)

        self.logger.info(f"sending job response: {reply}")
        self._set_edge_reply(reply, fl_ctx)
        fl_ctx.set_prop(FLContextKey.JOB_META, self.job_metas.get(job_id), private=True, sticky=False)

    @staticmethod
    def _set_edge_reply(reply, fl_ctx: FLContext):
        """Prepare the reply to the edge device.

        Args:
            reply: the reply to be set
            fl_ctx: FLContext object

        Returns: None

        """
        fl_ctx.set_prop(
            key=EdgeContextKey.REPLY_TO_EDGE,
            value=reply,
            private=True,
            sticky=False,
        )

    def _handle_edge_request(
        self,
        event_type: str,
        fl_ctx: FLContext,
        msg_topic: str,
        bad_req_reply,
        no_job_reply,
        comm_err_reply,
    ):
        self.logger.info(f"handling event {event_type}")
        req = fl_ctx.get_prop(EdgeContextKey.REQUEST_FROM_EDGE)
        job_id = req.job_id

        # try to find the job
        if not job_id:
            self.logger.error(f"handling event {event_type}: missing job_id from {type(req)}")
            self._set_edge_reply(bad_req_reply, fl_ctx)
            return

        if not self._find_job(job_id):
            self.logger.info(f"handling event {event_type}: no job found for {job_id}")
            self._set_edge_reply(no_job_reply, fl_ctx)
            return

        # send task request data to CJ
        self.logger.debug(f"Sending edge request to CJ {job_id}")
        engine = fl_ctx.get_engine()
        start = time.time()
        reply = engine.send_to_job(
            job_id=job_id,
            channel=CellChannel.EDGE_REQUEST,
            topic=msg_topic,
            msg=new_cell_message({}, req),
            timeout=self.request_timeout,
            optional=True,
        )

        assert isinstance(reply, CellMessage)
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)

        if rc != CellReturnCode.OK:
            self.logger.debug(f"Failed to get edge response after {time.time() - start} secs: {rc}")
            reply = comm_err_reply
        else:
            reply = reply.payload

        self._set_edge_reply(reply, fl_ctx)

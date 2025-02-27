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
from nvflare.edge.constants import EdgeContextKey, EdgeProtoKey
from nvflare.edge.constants import EventType as EdgeEventType
from nvflare.edge.constants import Status as EdgeStatus
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.widgets.widget import Widget


class EdgeTaskDispatcher(Widget):
    """Edge Task Dispatcher (ETD) is to be used to dispatch a received edge request to a running job (CJ).
    ETD must be installed on CP before the CP is started.
    """

    def __init__(self, request_timeout: float = 2.0):
        Widget.__init__(self)
        self.request_timeout = request_timeout
        self.edge_jobs = {}  # edge_method => list of job_ids
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

    def _remove_job(self, job_meta: dict):
        with self.lock:
            job_id = job_meta.get(JobMetaKey.JOB_ID)
            edge_method = job_meta.get(JobMetaKey.EDGE_METHOD)
            if not edge_method:
                # this is not an edge job
                self.logger.info(f"no edge_method in job {job_id}")
                return

            jobs = self.edge_jobs.get(edge_method)
            if not jobs:
                self.logger.info("no edge jobs pending")
                return

            assert isinstance(jobs, list)
            job_id = job_meta.get(JobMetaKey.JOB_ID)
            self.logger.info(f"current jobs for {edge_method}: {jobs}")
            if job_id in jobs:
                jobs.remove(job_id)
                if not jobs:
                    # no more jobs for this edge method
                    self.edge_jobs.pop(edge_method)

    def _match_job(self, caps: list):
        with self.lock:
            for edge_method, jobs in self.edge_jobs.items():
                if edge_method in caps:
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
        job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        if not job_meta:
            self.logger.error(f"missing {FLContextKey.JOB_META} from fl_ctx for event {event_type}")
        else:
            self._remove_job(job_meta)

    def _handle_edge_job_request(self, event_type: str, fl_ctx: FLContext):
        self.logger.info(f"handling event {event_type}")
        edge_capabilities = fl_ctx.get_prop(EdgeContextKey.EDGE_CAPABILITIES)
        if not edge_capabilities:
            self.logger.error(f"missing {EdgeContextKey.EDGE_CAPABILITIES} from fl_ctx for event {event_type}")
            self._set_edge_reply(EdgeStatus.INVALID_REQUEST, None, fl_ctx)
            return

        # find job for the caps
        job_id = self._match_job(edge_capabilities)
        if job_id:
            status = EdgeStatus.OK
        else:
            status = EdgeStatus.NO_JOB
        self._set_edge_reply(status, job_id, fl_ctx)

    @staticmethod
    def _set_edge_reply(status, data, fl_ctx: FLContext):
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
            self._set_edge_reply(EdgeStatus.INVALID_REQUEST, None, fl_ctx)
            return

        if not self._find_job(job_id):
            self._set_edge_reply(EdgeStatus.NO_JOB, None, fl_ctx)
            return

        # send edge request data to CJ
        edge_req_data = fl_ctx.get_prop(EdgeContextKey.REQUEST_FROM_EDGE)
        self.logger.info(f"Sending edge request to CJ {job_id}: {edge_req_data}")
        engine = fl_ctx.get_engine()
        reply = engine.send_to_job(
            job_id=job_id,
            channel=CellChannel.EDGE_REQUEST,
            topic="request",
            msg=new_cell_message({}, edge_req_data),
            timeout=self.request_timeout,
        )

        assert isinstance(reply, CellMessage)
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        reply_data = reply.payload
        self.logger.debug(f"got edge result from CJ: {rc=} {reply_data=}")
        self._set_edge_reply(rc, reply_data, fl_ctx)

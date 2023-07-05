# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""The FedAdmin to communicate with the Admin server."""

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.cell import Message as CellMessage
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.fuel.sec.authz import AuthorizationService, AuthzContext, Person
from nvflare.private.admin_defs import Message, error_reply, ok_reply
from nvflare.private.defs import CellChannel, RequestHeader, new_cell_message
from nvflare.security.logging import secure_format_exception, secure_log_traceback


class RequestProcessor(object):
    """The RequestProcessor is responsible for processing a request."""

    def get_topics(self) -> [str]:
        """Get topics that this processor will handle.

        Returns: list of topics

        """
        pass

    def process(self, req: Message, app_ctx) -> Message:
        """Called to process the specified request.

        Args:
            req: request message
            app_ctx: application context

        Returns: reply message

        """
        pass


class FedAdminAgent(object):
    """FedAdminAgent communicate with the FedAdminServer."""

    def __init__(self, client_name: str, cell: Cell, app_ctx):
        """Init the FedAdminAgent.

        Args:
            client_name: client name
            app_ctx: application context
            cell: the Cell for communication
        """
        auditor = AuditService.get_auditor()
        if not isinstance(auditor, Auditor):
            raise TypeError("auditor must be an instance of Auditor, but got {}".format(type(auditor)))

        self.name = client_name
        self.cell = cell
        self.auditor = auditor
        self.app_ctx = app_ctx
        self.processors = {}
        self.asked_to_stop = False
        self.register_cell_cb()

    def register_cell_cb(self):
        self.cell.register_request_cb(
            channel=CellChannel.CLIENT_MAIN,
            topic="*",
            cb=self._dispatch_request,
        )

    def register_processor(self, processor: RequestProcessor):
        """To register the RequestProcessor.

        Args:
            processor: RequestProcessor

        """
        if not isinstance(processor, RequestProcessor):
            raise TypeError("processor must be an instance of RequestProcessor, but got {}".format(type(processor)))

        topics = processor.get_topics()
        for topic in topics:
            assert topic not in self.processors, "duplicate processors for topic {}".format(topic)
            self.processors[topic] = processor

    def _dispatch_request(
        self,
        request: CellMessage,
        # *args, **kwargs
    ) -> CellMessage:
        assert isinstance(request, CellMessage), "request must be CellMessage but got {}".format(type(request))
        req = request.payload

        assert isinstance(req, Message), "request payload must be Message but got {}".format(type(req))
        topic = req.topic

        # create audit record
        if self.auditor:
            user_name = req.get_header(RequestHeader.USER_NAME, "")
            ref_event_id = req.get_header(ConnProps.EVENT_ID, "")
            self.auditor.add_event(user=user_name, action=topic, ref=ref_event_id)

        processor: RequestProcessor = self.processors.get(topic)
        if processor:
            try:
                reply = None

                # see whether pre-authorization is needed
                authz_flag = req.get_header(RequestHeader.REQUIRE_AUTHZ)
                require_authz = authz_flag == "true"
                if require_authz:
                    # authorize this command!
                    cmd = req.get_header(RequestHeader.ADMIN_COMMAND, None)
                    if cmd:
                        user = Person(
                            name=req.get_header(RequestHeader.USER_NAME, ""),
                            org=req.get_header(RequestHeader.USER_ORG, ""),
                            role=req.get_header(RequestHeader.USER_ROLE, ""),
                        )
                        submitter = Person(
                            name=req.get_header(RequestHeader.SUBMITTER_NAME, ""),
                            org=req.get_header(RequestHeader.SUBMITTER_ORG, ""),
                            role=req.get_header(RequestHeader.SUBMITTER_ROLE, ""),
                        )

                        authz_ctx = AuthzContext(user=user, submitter=submitter, right=cmd)
                        authorized, err = AuthorizationService.authorize(authz_ctx)
                        if err:
                            reply = error_reply(err)
                        elif not authorized:
                            reply = error_reply("not authorized")
                    else:
                        reply = error_reply("requires authz but missing admin command")

                if not reply:
                    reply = processor.process(req, self.app_ctx)
                    if reply is None:
                        # simply ack
                        reply = ok_reply()
                    else:
                        if not isinstance(reply, Message):
                            raise RuntimeError(f"processor for topic {topic} failed to produce valid reply")
            except Exception as e:
                secure_log_traceback()
                reply = error_reply(f"exception_occurred: {secure_format_exception(e)}")
        else:
            reply = error_reply("invalid_request")
        return new_cell_message({}, reply)

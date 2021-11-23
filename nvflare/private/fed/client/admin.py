# Copyright (c) 2021, NVIDIA CORPORATION.
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
import traceback

from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.private.admin_defs import Message, error_reply, ok_reply


class Sender(object):
    """
    The Sender object integrate the agent with the underline messaging system.
    Make sure its methods are exception-proof!
    """

    def send_reply(self, reply: Message):
        """
        Send the reply to the requester.
        :param reply:
        :return:
        """
        pass

    def retrieve_requests(self) -> [Message]:
        """
        Send the message to retrieve pending requests from the Server
        :return:
        """
        pass

    def send_result(self, message: Message):
        """
        Send the processor results to server.
        :param reply:
        :return:
        """
        pass


class RequestProcessor(object):
    """
    The RequestProcessor is responsible for processing a request.
    """

    def get_topics(self) -> [str]:
        """
        Get topics that this processor will handle
        :return: list of topics
        """
        pass

    def process(self, req: Message, app_ctx) -> Message:
        """
        Called to process the specified request
        :param req:
        :param app_ctx:
        :return: a reply message
        """
        pass


class FedAdminAgent(object):
    def __init__(self, client_name, sender: Sender, app_ctx, req_poll_interval=0.5, process_poll_interval=0.1):
        assert isinstance(sender, Sender), "sender must be Sender"

        auditor = AuditService.get_auditor()
        assert isinstance(auditor, Auditor), "auditor must be Auditor but got {}".format(type(auditor))

        self.name = client_name
        self.sender = sender
        self.auditor = auditor
        self.app_ctx = app_ctx
        self.req_poll_interval = req_poll_interval
        self.process_poll_interval = process_poll_interval
        self.processors = {}
        self.reqs = []
        self.req_lock = threading.Lock()
        self.retrieve_reqs_thread = None
        self.process_req_thread = None
        self.asked_to_stop = False

    def register_processor(self, processor: RequestProcessor):
        assert isinstance(processor, RequestProcessor), "processor must be RequestProcessor"

        topics = processor.get_topics()
        for topic in topics:
            assert topic not in self.processors, "duplicate processors for topic {}".format(topic)
            self.processors[topic] = processor

    def start(self):
        if self.retrieve_reqs_thread is None:
            self.retrieve_reqs_thread = threading.Thread(target=_start_retriever, args=(self,))

        # called from the main thread
        if not self.retrieve_reqs_thread.is_alive():
            self.retrieve_reqs_thread.start()

        if self.process_req_thread is None:
            self.process_req_thread = threading.Thread(target=_start_processor, args=(self,))

        # called from the main thread
        if not self.process_req_thread.is_alive():
            self.process_req_thread.start()

    def _run_retriever(self):
        while True:
            if self.asked_to_stop:
                break

            reqs = self.sender.retrieve_requests()
            if reqs is not None and isinstance(reqs, list):
                with self.req_lock:
                    self.reqs.extend(reqs)

            time.sleep(self.req_poll_interval)

    def _run_processor(self):
        while True:
            if self.asked_to_stop:
                break

            with self.req_lock:
                if len(self.reqs) > 0:
                    req = self.reqs.pop(0)
                else:
                    req = None

            if req:
                assert isinstance(req, Message), "request must be Message but got {}".format(type(req))
                topic = req.topic

                # create audit record
                if self.auditor:
                    user_name = req.get_header(ConnProps.USER_NAME, "")
                    ref_event_id = req.get_header(ConnProps.EVENT_ID, "")
                    self.auditor.add_event(user=user_name, action=topic, ref=ref_event_id)

                processor = self.processors.get(topic)
                if processor:
                    try:
                        reply = processor.process(req, self.app_ctx)
                        if reply is None:
                            # simply ack
                            reply = ok_reply()
                        else:
                            assert isinstance(
                                reply, Message
                            ), "processor for topic {} failed to produce valid reply".format(topic)
                    except BaseException as e:
                        traceback.print_exc()
                        reply = error_reply("exception_occurred")
                else:
                    reply = error_reply("invalid_request")

                reply.set_ref_id(req.id)
                self.sender.send_reply(reply)

            time.sleep(self.process_poll_interval)

    def shutdown(self):
        """
        To be called by the Client Engine to gracefully shutdown the agent.
        :return:
        """
        self.asked_to_stop = True

        if self.retrieve_reqs_thread and self.retrieve_reqs_thread.is_alive():
            self.retrieve_reqs_thread.join()

        if self.process_req_thread and self.process_req_thread.is_alive():
            self.process_req_thread.join()


def _start_retriever(agent: FedAdminAgent):
    agent._run_retriever()


def _start_processor(agent: FedAdminAgent):
    agent._run_processor()

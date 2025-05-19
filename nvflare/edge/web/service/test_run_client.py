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
import logging
import threading
import time

from nvflare.edge.web.models.capabilities import Capabilities
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.user_info import UserInfo
from nvflare.edge.web.service.client import EdgeApiClient, Reply
from nvflare.edge.web.service.utils import grpc_reply_to_job_response, job_request_to_grpc_request


class ReqInfo:

    def __init__(self, idx):
        self.idx = idx
        self.send_time = None
        self.rcv_time = None


def request_job(req: ReqInfo, client: EdgeApiClient, addr):
    req.send_time = time.time()
    print(f"{req.idx}: sending job request")

    job_req = JobRequest(
        device_info=DeviceInfo("aaaaa"),
        user_info=UserInfo(user_name="john"),
        capabilities=Capabilities(["xgb", "deep-learn"]),
    )
    request = job_request_to_grpc_request(job_req)
    reply = client.query(addr, request)
    req.rcv_time = time.time()
    assert isinstance(reply, Reply)
    resp = grpc_reply_to_job_response(reply)
    print(f"{req.idx}: got response after {req.rcv_time - req.send_time} seconds: {resp}")


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    client = EdgeApiClient()
    addr = "127.0.0.1:8009"

    num_threads = 5
    reqs = []
    for i in range(num_threads):
        req = ReqInfo(i + 1)
        reqs.append(req)
        t = threading.Thread(target=request_job, args=(req, client, addr), daemon=True)
        t.start()

    while True:
        all_done = True
        for r in reqs:
            if not r.rcv_time:
                all_done = False
                break
        if all_done:
            break

    max_duration = 0
    for r in reqs:
        d = r.rcv_time - r.send_time
        if max_duration < d:
            max_duration = d

    print(f"{time.time()}: ALL DONE! {max_duration=}")


if __name__ == "__main__":
    main()

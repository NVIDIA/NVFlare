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
import json
import logging
import threading
import time

from nvflare.edge.constants import HttpHeaderKey
from nvflare.edge.web.grpc.client import Client, Reply, Request
from nvflare.edge.web.grpc.constants import QueryType


class ReqInfo:

    def __init__(self, idx):
        self.idx = idx
        self.send_time = None
        self.rcv_time = None


def _to_bytes(d: dict) -> bytes:
    str_data = json.dumps(d)
    return str_data.encode("utf-8")


def request_job(req: ReqInfo, client: Client, addr):
    req.send_time = time.time()
    print(f"{req.idx}: sending job request")
    header = {
        HttpHeaderKey.DEVICE_ID: "aaaaa",
    }

    job_payload = {"capabilities": {"methods": ["xgb", "deep-learn"]}}

    request = Request(
        type=QueryType.JOB_REQUEST,
        method="post",
        header=_to_bytes(header),
        payload=_to_bytes(job_payload),
    )
    reply = client.query(addr, request)
    req.rcv_time = time.time()
    print(f"{req.idx}: got reply after {req.rcv_time - req.send_time} seconds")
    assert isinstance(reply, Reply)


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    client = Client()
    addr = "127.0.0.1:8009"

    num_threads = 30
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

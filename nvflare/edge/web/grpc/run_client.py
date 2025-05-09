import json
import logging
import threading
import time
import uuid

from nvflare.edge.web.grpc.client import Client, Request, Reply
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.selection_response import SelectionResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.web.grpc.constants import NONE_DATA, QueryType
from nvflare.edge.constants import HttpHeaderKey


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

    job_payload = {
        "capabilities": {"methods": ["xgb", "deep-learn"]}
    }

    request = Request(
        type=QueryType.JOB_REQUEST,
        method="post",
        header=_to_bytes(header),
        payload=_to_bytes(job_payload),
        cookie=NONE_DATA,
    )
    reply = client.query(addr, request)
    req.rcv_time = time.time()
    print(f"{req.idx}: got reply after {req.rcv_time-req.send_time} seconds")
    assert isinstance(reply, Reply)


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    client = Client()
    addr = "127.0.0.1:8009"

    num_threads = 90
    reqs = []
    for i in range(num_threads):
        req = ReqInfo(i+1)
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

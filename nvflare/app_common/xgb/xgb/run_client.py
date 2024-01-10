import time

from nvflare.app_common.xgb.xgb.client import XGBClient
import nvflare.app_common.xgb.proto.federated_pb2 as pb2


import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", "-a", type=str, help="server address", required=True)
    parser.add_argument("--rank", "-r", type=int, help="client rank", required=True)

    args = parser.parse_args()
    client = XGBClient(server_addr=args.addr)
    client.start()

    rank = args.rank
    seq = 0
    total_time = 0
    total_reqs = 0
    for i in range(10):
        print(f"Test round {i}")
        data = os.urandom(1000000)

        print("sending allgather")
        start = time.time()
        result = client.send_allgather(
            seq_num=seq+1,
            rank=rank,
            data=data
        )
        total_reqs += 1
        total_time += time.time() - start
        if not isinstance(result, pb2.AllgatherReply):
            print(f"expect reply to be pb2.AllgatherReply but got {type(result)}")
        elif result.receive_buffer != data:
            print("ERROR: allgather result does not match request")
        else:
            print("OK: allgather result matches request!")

        print("sending allgatherV")
        start = time.time()
        result = client.send_allgatherv(
            seq_num=seq + 2,
            rank=rank,
            data=data
        )
        total_reqs += 1
        total_time += time.time() - start
        if not isinstance(result, pb2.AllgatherVReply):
            print(f"expect reply to be pb2.AllgatherVReply but got {type(result)}")
        elif result.receive_buffer != data:
            print("ERROR: allgatherV result does not match request")
        else:
            print("OK: allgatherV result matches request!")

        print("sending allreduce")
        start = time.time()
        result = client.send_allreduce(
            seq_num=seq + 3,
            rank=rank,
            data=data,
            reduce_op=2,
            data_type=2,
        )
        total_reqs += 1
        total_time += time.time() - start
        if not isinstance(result, pb2.AllreduceReply):
            print(f"expect reply to be pb2.AllreduceReply but got {type(result)}")
        elif result.receive_buffer != data:
            print("ERROR: allreduce result does not match request")
        else:
            print("OK: allreduce result matches request!")

        print("sending broadcast")
        start = time.time()
        result = client.send_broadcast(
            seq_num=seq + 4,
            rank=rank,
            data=data,
            root=3,
        )
        total_reqs += 1
        total_time += time.time() - start
        if not isinstance(result, pb2.BroadcastReply):
            print(f"expect reply to be pb2.BroadcastReply but got {type(result)}")
        elif result.receive_buffer != data:
            print("ERROR: broadcast result does not match request")
        else:
            print("OK: broadcast result matches request!")

        seq += 4
        time.sleep(1.0)

    time_per_req = total_time / total_reqs
    print(f"DONE: {total_reqs=} {total_time=} {time_per_req=}")


if __name__ == "__main__":
    main()

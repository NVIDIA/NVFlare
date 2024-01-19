import argparse
import logging

from nvflare.app_common.xgb.bridges.grpc.server import XGBServer
from nvflare.app_common.xgb.bridges.grpc.mock.aggr_servicer import AggrServicer


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", "-a", type=str, help="server address", required=True)
    parser.add_argument("--num_clients", "-c", type=int, help="number of clients", required=True)
    parser.add_argument("--max_workers", "-w", type=int, help="max number of workers", required=False, default=20)

    args = parser.parse_args()
    print(f"starting server at {args.addr} max_workers={args.max_workers}")
    server = XGBServer(
        args.addr,
        max_workers=args.max_workers,
        options=None,
        servicer=AggrServicer(num_clients=args.num_clients),
    )
    server.start()


if __name__ == "__main__":
    main()

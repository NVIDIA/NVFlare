import argparse
import logging
import threading
import time

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.cellnet.net_manager import NetManager
from nvflare.fuel.f3.mpm import MainProcessMonitor


class Worker:
    def __init__(self, fqcn: str, root_url, start_time: float):
        self.cell = None
        self.fqcn = fqcn
        self.root_url = root_url
        self.waiter = threading.Event()
        self.start_time = start_time
        MainProcessMonitor.add_cleanup_cb(self._cleanup)

    def _cleanup(self):
        if self.cell:
            self.stop()

    def start(self):
        start = time.time()
        self.cell = Cell(
            fqcn=self.fqcn,
            root_url=self.root_url,
            secure=False,
            credentials={},
        )
        _ = NetAgent(self.cell, agent_closed_cb=self.stop)
        self.cell.start()
        while not self.cell.is_cell_connected(FQCN.ROOT_SERVER):
            time.sleep(0.001)
        print(f"Cell {self.fqcn}: establish time: {time.time()-start} seconds")
        print(f"Cell {self.fqcn}: since process start time: {time.time()-self.start_time} seconds")
        self.waiter.wait()

    def stop(self):
        self.waiter.set()


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--time", "-t", type=float, help="start time", required=True)
    parser.add_argument("--name", "-n", type=str, help="my cell name", required=True)
    parser.add_argument("--root_url", "-r", type=str, help="root url", required=True)
    args = parser.parse_args()

    print(f"Starting worker {args.name} connect to {args.root_url}")
    worker = Worker(args.name, args.root_url, args.time)
    worker.start()


if __name__ == "__main__":
    MainProcessMonitor.run(main)

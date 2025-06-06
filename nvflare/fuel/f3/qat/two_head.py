import os
import shlex
import subprocess
import sys
import threading
import time

from nvflare.fuel.f3.cellnet.core_cell import CellAgent, CoreCell, Message, MessageHeaderKey, MessageType
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.mpm import MainProcessMonitor


class TwoHeader:
    def __init__(self, root1: str, name1: str, root2: str, name2: str):
        self.waiter = threading.Event()

        self.cell1 = CoreCell(
            fqcn=name1,
            root_url=root1,
            secure=False,
            credentials={},
            create_internal_listener=False,
        )
        self.agent1 = NetAgent(self.cell1, agent_closed_cb=self._agent1_closed)

        self.cell2 = CoreCell(
            fqcn=name2,
            root_url=root2,
            secure=False,
            credentials={},
            create_internal_listener=False,
        )
        self.agent2 = NetAgent(self.cell2, agent_closed_cb=self._agent2_closed)

    def _agent1_closed(self):
        print("cell1 stopped")
        self.cell1.stop()
        self.cell1_stopped = True
        self.stop()

    def _agent2_closed(self):
        print("cell2 stopped")
        self.cell2.stop()
        self.cell2_stopped = True
        self.stop()

    def start(self):
        self.cell1.start()
        self.cell2.start()

    def stop(self):
        if self.cell1_stopped and self.cell2_stopped:
            print("Both stopped!")
            self.waiter.set()

    def run(self):
        self.waiter.wait()


def main():
    root1 = "grpc://localhost:8002"
    name1 = "h1"

    root2 = "grpc://localhost:9002"
    name2 = "h2"

    h = TwoHeader(
        root1=root1,
        name1=name1,
        root2=root2,
        name2=name2,
    )

    h.start()
    h.run()


if __name__ == "__main__":
    MainProcessMonitor.run(main)

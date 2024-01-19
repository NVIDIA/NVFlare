import subprocess
import shlex
import os
import threading
import signal

from nvflare.fuel.utils.obj_utils import get_logger


class ProcessManager:

    def __init__(self, name: str, start_cmd: str):
        self.name = name
        self.start_cmd = start_cmd
        self.process = None
        self.logger = get_logger(self)
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if self.process:
                return
            self.logger.info(f"starting process {self.name}: {self.start_cmd}")
            self.process = subprocess.Popen(
                shlex.split(self.start_cmd), preexec_fn=os.setsid,
                env=os.environ.copy())
            self.logger.info(f"started process {self.name}: {self.start_cmd}")

    def is_stopped(self) -> (bool, int):
        with self.lock:
            if not self.process:
                return True, 0

            rc = self.process.poll()
            if rc is None:
                # still running
                return False, 0
            else:
                # stopped
                self.process = None
                self.logger.info(f"process {self.name} is stopped with RC {rc}")
                return True, rc

    def stop(self):
        with self.lock:
            if self.process:
                # kill the process
                self.logger.info(f"stopping process {self.name}")
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    pass
                self.process = None

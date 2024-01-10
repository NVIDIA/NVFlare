import subprocess
import shlex
import os
import threading
import signal
import time

from nvflare.fuel.utils.obj_utils import get_logger


class ProcessManager:

    def __init__(self, name: str, start_cmd: str, stopped_cb, **cb_kwargs):
        self.name = name
        self.start_cmd = start_cmd
        self.stopped_cb = stopped_cb
        self.cb_kwargs = cb_kwargs
        self.process = None
        self.asked_to_stop = False
        self.logger = get_logger(self)

    def start(self):
        self.logger.info(f"starting process {self.name}: {self.start_cmd}")
        self.process = subprocess.Popen(
            shlex.split(self.start_cmd), preexec_fn=os.setsid,
            env=os.environ.copy())
        monitor = threading.Thread(target=self._monitor_process, daemon=True)
        monitor.start()
        self.logger.info(f"started process {self.name}: {self.start_cmd}")

    def _monitor_process(self):
        while not self.asked_to_stop:
            rc = self.process.poll()
            if rc is None:
                # still running
                time.sleep(1.0)
            else:
                # done
                self.logger.info(f"process {self.name} is stopped with RC {rc}")
                self.process = None
                if self.stopped_cb is not None:
                    self.stopped_cb(rc, **self.cb_kwargs)
                break
        if self.process:
            # kill the process
            self.logger.info(f"killed process {self.name}")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except:
                pass

    def stop(self):
        self.asked_to_stop = True

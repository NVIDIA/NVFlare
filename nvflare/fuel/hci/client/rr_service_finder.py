# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from .api_spec import ServiceFinder


class RRServiceFinder(ServiceFinder):
    def __init__(self, change_interval, host1: str, port1: int, host2: str, port2: int):
        self.host1 = host1
        self.port1 = port1
        self.host2 = host2
        self.port2 = port2
        self.change_interval = change_interval

        self.thread = None
        self.stop_asked = False

    def start(self, service_address_changed_cb):
        self.thread = threading.Thread(target=self._gen_address, args=(service_address_changed_cb,), daemon=True)
        self.thread.start()

    def _gen_address(self, service_address_changed_cb):
        last_port = self.port1
        last_change_time = None
        while True:
            if self.stop_asked:
                return

            if not last_change_time or time.time() - last_change_time >= self.change_interval:
                last_change_time = time.time()
                if last_port == self.port1:
                    h = self.host2
                    p = self.port2
                else:
                    h = self.host1
                    p = self.port1

                service_address_changed_cb(h, p, "1234")
                last_port = p
            time.sleep(0.2)

    def stop(self):
        self.stop_asked = True
        if self.thread and self.thread.is_alive():
            self.thread.join()
        print("Service finder stopped")

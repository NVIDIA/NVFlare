# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Server-side SplitNN workflow expressed as direct CollabAPI calls."""


from nvflare.collab import collab
from nvflare.fuel.utils.log_utils import get_obj_logger

RUN_TIMEOUT = 7_200.0


class SplitNNServer:
    """Server-side owner of the Collab workflow entry point."""

    def __init__(self):
        self.logger = get_obj_logger(self)

    def _clients(self):
        # collab.clients exposes the remote client proxies participating in
        # this run; job.py gives these two proxies stable role assignments.
        clients = {client.name: client for client in collab.clients}
        missing = sorted({"site-1", "site-2"} - clients.keys())
        if missing:
            raise RuntimeError(f"SplitNN requires site-1 and site-2; missing {missing}")
        return clients["site-1"], clients["site-2"]

    # Every Collab server defines exactly one @collab.main method. NVFlare
    # invokes it after the configured clients have initialized.
    @collab.main
    def run(self):
        image_client, _ = self._clients()
        self.logger.info("starting SplitNN coordinator on image-side site-1")
        # Calling the proxy invokes site-1's published run_splitnn method.
        # Its normal Python dict return value becomes the workflow result.
        return image_client(timeout=RUN_TIMEOUT).run_splitnn()

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

"""Server-side SplitNN workflow expressed as direct Collab API calls."""


from nvflare.collab import collab
from nvflare.fuel.utils.log_utils import get_obj_logger

RUN_TIMEOUT = 86_400.0


class SplitNNServer:
    """Server-side owner of the Collab workflow entry point."""

    def __init__(self, image_site: str):
        self.image_site = image_site
        self.logger = get_obj_logger(self)

    # Every Collab server defines exactly one @collab.main method. NVFlare
    # invokes it after the configured clients have initialized.
    @collab.main
    def run(self):
        # get_clients validates that the configured image site participated
        # and returns its callable proxy.
        image_client = collab.get_clients([self.image_site])[0]
        self.logger.info(f"starting SplitNN coordinator on image-side {self.image_site}")
        # Calling the proxy invokes the image site's published run_splitnn method.
        # Its normal Python dict return value becomes the workflow result.
        return image_client(timeout=RUN_TIMEOUT).run_splitnn()

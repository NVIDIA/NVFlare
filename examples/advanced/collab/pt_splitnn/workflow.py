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


class SplitNNWorkflow:
    def __init__(
        self,
        train_size: int,
        test_size: int,
        num_steps: int = 15625,
        batch_size: int = 64,
        validation_frequency: int = 1000,
        call_timeout: float = 600.0,
        seed: int = 42,
        run_timeout: float = 7200.0,
        log_frequency: int = 100,
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.validation_frequency = validation_frequency
        self.call_timeout = call_timeout
        self.seed = seed
        self.log_frequency = log_frequency
        self.logger = get_obj_logger(self)

        self.run_timeout = run_timeout

    def _clients(self):
        clients = {client.name: client for client in collab.clients}
        missing = sorted({"site-1", "site-2"} - clients.keys())
        if missing:
            raise RuntimeError(f"SplitNN requires site-1 and site-2; missing {missing}")
        return clients["site-1"], clients["site-2"]

    @collab.main
    def run(self):
        image_client, _ = self._clients()
        self.logger.info(
            f"starting image-side SplitNN coordinator for {self.num_steps} steps with batch size {self.batch_size}"
        )
        return image_client(timeout=self.run_timeout).run_splitnn(
            train_size=self.train_size,
            test_size=self.test_size,
            num_steps=self.num_steps,
            batch_size=self.batch_size,
            validation_frequency=self.validation_frequency,
            call_timeout=self.call_timeout,
            seed=self.seed,
            log_frequency=self.log_frequency,
        )

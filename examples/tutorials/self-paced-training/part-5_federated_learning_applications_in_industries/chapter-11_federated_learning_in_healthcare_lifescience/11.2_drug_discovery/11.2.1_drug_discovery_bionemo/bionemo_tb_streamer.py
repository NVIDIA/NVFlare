# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from pytorch_lightning.callbacks import Callback

from nvflare.client.tracking import SummaryWriter


class BioNeMoTBStreamer(Callback):
    def __init__(self, start_step=0):
        """FL callback for lightning API.

        Args:
            start_step: current step to start logging. Any subsequent validation call will increment the step by 1.
        """
        super().__init__()
        self.start_step = start_step
        self.current_step = start_step
        self.summary_writer = SummaryWriter()

    def on_validation_end(self, trainer, pl_module):
        self._stream_metrics(trainer.logged_metrics)
        self.current_step += 1

    def _stream_metrics(self, metrics):
        for k, v in metrics.items():
            self.summary_writer.add_scalar(tag=k, scalar=v.item(), global_step=self.current_step)

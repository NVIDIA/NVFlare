# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from nvflare.app_opt.lightning.fl_callbacks import FLCallback


def patch(trainer: pl.Trainer):
    fl_callback = FLCallback()
    callbacks = trainer.callbacks
    if isinstance(callbacks, list):
        callbacks.append(fl_callback)
    elif isinstance(callbacks, Callback):
        callbacks = [callbacks, fl_callback]
    else:
        callbacks = [fl_callback]
    trainer.callbacks = callbacks

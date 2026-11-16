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

import numpy as np
from model import train

from nvflare.collab import collab


class Trainer:
    @collab.publish
    def train(self, weights):
        # Simulate a different local dataset at each site. Only updates leave the site.
        rng = np.random.default_rng(collab.get_app_prop("seed"))
        count = collab.get_app_prop("samples")
        x = rng.normal(size=(count, 3))
        y = x @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=count)
        weights, loss = train(weights, x, y, epochs=5)
        return {"weights": weights, "samples": count, "loss": loss}

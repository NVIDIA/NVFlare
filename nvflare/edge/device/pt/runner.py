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
from nvflare.edge.device.sdk_spec import FlareRunner

from .optimizers import AdamOptimizerWrapper, SGDOptimizerWrapper
from .trainer import PTTrainer


class PTRunner(FlareRunner):

    def add_builtin_resolvers(self):
        self.resolver_registry.update(
            {
                "Trainer.DLTrainer": PTTrainer,
                "Optimizer.SGD": SGDOptimizerWrapper,
                "Optimizer.Adam": AdamOptimizerWrapper,
            }
        )

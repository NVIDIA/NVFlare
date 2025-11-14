# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any, Optional

from torchvision.models import ResNet
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.resnet import BasicBlock, ResNet18_Weights


class Resnet18(ResNet):
    def __init__(self, num_classes, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any):
        self.num_classes = num_classes

        weights = ResNet18_Weights.verify(weights)

        if weights is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

        if weights is not None:
            super().load_state_dict(weights.get_state_dict(progress=progress))

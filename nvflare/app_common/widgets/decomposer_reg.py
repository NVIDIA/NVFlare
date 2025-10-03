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
from typing import List

from nvflare.fuel.utils.class_loader import load_class
from nvflare.fuel.utils.fobs import register
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.widgets.widget import Widget


class DecomposerRegister(Widget):

    def __init__(self, classes: List[str]):
        self.classes = classes
        logger = get_obj_logger(self)
        Widget.__init__(self)
        for class_name in classes:
            register(load_class(class_name))
            logger.info(f"Registered decomposer {class_name}")

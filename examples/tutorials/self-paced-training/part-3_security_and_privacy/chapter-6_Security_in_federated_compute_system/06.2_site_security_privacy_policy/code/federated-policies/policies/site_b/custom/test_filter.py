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
import logging

from nvflare.apis.filter import ContentBlockedException, Filter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable

log = logging.getLogger(__name__)


class TestFilter(Filter):
    def __init__(self, name, block=False):
        self.name = name
        self.block = block

    def process(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        if self.block:
            log.info(f"Filter {self.name} blocked the content")
            raise ContentBlockedException("Content blocked by filter " + self.name)

        log.info(f"Filter {self.name} is invoked")
        return shareable

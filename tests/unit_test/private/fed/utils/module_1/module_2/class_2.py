# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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


class Class2:
    def __init__(self):
        self.LOG = logging.getLogger(__name__)

    def info_method(self):
        self.LOG.info("this is info in module_1.module_2.Class2")

    def debug_method(self):
        self.LOG.debug("this is debug in module_1.module_2.Class2")

    def warning_method(self):
        self.LOG.warning("this is warning in module_1.module_2.Class2")

    def error_method(self):
        self.LOG.error("this is error in module_1.module_2.Class2")

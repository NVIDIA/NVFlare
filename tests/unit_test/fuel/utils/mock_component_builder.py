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
from nvflare.fuel.utils.class_utils import ModuleScanner
from nvflare.fuel.utils.component_builder import ComponentBuilder


class MockComponentBuilder(ComponentBuilder):
    def __init__(self):
        self.module_scanner = ModuleScanner(["nvflare"], ["api", "app_commons", "app_opt", "fuel", "private", "utils"])

    def get_module_scanner(self):
        return self.module_scanner

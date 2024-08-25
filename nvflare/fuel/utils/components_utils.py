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
import json
import logging
import os

COMPONENT_CLASS_FILE = "component_classes.json"
logger = logging.getLogger(__name__)


def create_classes_table_static():
    class_table = {}
    try:
        file = os.path.join(os.path.dirname(__file__), COMPONENT_CLASS_FILE)
        with open(file, "r") as f:
            class_table = json.load(f)
    except Exception as ex:
        logger.warning(f"Exception occurred when loading class table from {file}: {ex}")
    return class_table


if __name__ == "__main__":

    from nvflare.fuel.utils.class_utils import ModuleScanner

    module_scanner = ModuleScanner(["nvflare"], ["apis", "app_common", "app_opt", "widgets"], True)
    class_table = module_scanner.create_classes_table()

    file = os.path.join(os.path.dirname(__file__), COMPONENT_CLASS_FILE)
    json_object = json.dumps(class_table, indent=4)
    with open(file, "w") as f:
        f.write(json_object)

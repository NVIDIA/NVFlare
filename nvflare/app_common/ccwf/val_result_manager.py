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

import copy
import os
import threading

from nvflare.apis.dxo import DXO


class EvalResultManager:
    def __init__(self, result_dir: str):
        self.result_dir = result_dir
        self.results = {}
        self.update_lock = threading.Lock()
        if not os.path.exists(result_dir):
            # create
            os.makedirs(result_dir)

    def add_result(self, evaluator: str, evaluatee: str, result: DXO):
        with self.update_lock:
            save_file_name = evaluator + "-" + evaluatee
            file_path = self._save_validation_result(save_file_name, result)

            if evaluator not in self.results:
                self.results[evaluator] = {}

            self.results[evaluator][evaluatee] = file_path
            return file_path

    def _save_validation_result(self, file_name, result: DXO):
        file_path = os.path.join(self.result_dir, file_name)
        result.to_file(file_path)
        return file_path

    def get_results(self):
        with self.update_lock:
            return copy.deepcopy(self.results)

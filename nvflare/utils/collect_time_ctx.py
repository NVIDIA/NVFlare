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


import timeit


class CollectTimeContext:
    def __init__(self):
        self.metrics = {"count": 0, "time_taken": 0.0, "error_count": 0}
        self.start_time = None

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = timeit.default_timer() - self.start_time

        if exc_type is not None:
            # An exception occurred
            self.metrics["error_count"] += 1
        else:
            # No exception occurred
            self.metrics["count"] += 1
            self.metrics["time_taken"] += elapsed_time
        # Return False to propagate the exception if there was one
        return False

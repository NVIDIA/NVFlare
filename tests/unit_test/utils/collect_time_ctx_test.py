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


# Example function that can raise an exception
import time

import pytest

from nvflare.utils.collect_time_ctx import CollectTimeContext


def example_function(param):
    print(f"Inside example_function with param: {param}")
    if param == "error":
        raise ValueError("Example error occurred.")
    time.sleep(0.5)


# Example command name extractor function
def extract_command_name(param):
    return f"example_function_param_{param}"


class TestDecorators:
    def test_collection_fn(self):
        # Example usage of CollectTimeContext with dynamic command_name_extractor
        values = ["cmd1", "cmd2", "cmd3", "cmd3"]
        for param_value in values:
            with CollectTimeContext() as context:
                example_function(param_value)
            metrics = context.metrics
            assert metrics["count"] == 1
            assert metrics["error_count"] == 0
            assert metrics["time_taken"] > 0.5
        with pytest.raises(Exception):
            with CollectTimeContext() as context:
                example_function("error")

            metrics = context.metrics
            assert metrics["count"] == 0
            assert metrics["error_count"] == 1
            assert metrics["time_taken"] == 0

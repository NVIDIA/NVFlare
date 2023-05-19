# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import time


def time_to_string(t) -> str:
    """Convert time into a formatted string.

    Args:
        t: input time string in seconds since the Epoch

    Returns:
        formatted time string
    """
    if t is None:
        return "N/A"

    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))

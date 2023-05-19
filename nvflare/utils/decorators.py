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

import functools
import timeit


def collect_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "reset" in kwargs and kwargs["reset"]:
            wrapper.time_taken = 0
            wrapper.count = 0
        else:
            start = timeit.default_timer()
            result = func(*args, **kwargs)
            wrapper.time_taken += (timeit.default_timer() - start) * 1000.0
            wrapper.count += 1
            return result

    wrapper.time_taken = 0
    wrapper.count = 0
    return wrapper


def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        duration = (timeit.default_timer() - start) * 1000.0
        wrapper.time_taken = duration
        return result

    return wrapper

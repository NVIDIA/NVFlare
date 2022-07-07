# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
import unittest

quick_test_var = "QUICKTEST"


def test_is_quick():
    return os.environ.get(quick_test_var, "").lower() == "true"


def skip_if_quick(obj):
    """
    Skip the unit tests if environment variable `quick_test_var=true`.
    For example, the user can skip the relevant tests by setting ``export QUICKTEST=true``.
    """
    is_quick = test_is_quick()

    return unittest.skipIf(is_quick, "Skipping slow tests")(obj)

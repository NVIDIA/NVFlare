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
from collections import OrderedDict

from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposers.core_decomposers import OrderedDictDecomposer


class TestDecomposers:
    def test_sorted_dict(self):

        test_list = [(3, "First"), (1, "Middle"), (2, "Last")]
        test_data = OrderedDict(test_list)

        buffer = fobs.dumps(test_data)
        new_data = fobs.loads(buffer)
        new_list = list(new_data.items())

        assert test_list == new_list

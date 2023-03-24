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

import re


def parse_summary(str):
    str = str.strip()
    str = re.sub("\n+", "\t", str)
    tmp_list = str.split("\t")
    # print(tmp_list)

    new_list = [re.split(r"  +", str.strip()) for str in tmp_list][1:]
    # print(new_list)

    ret_dict = {}
    for l in new_list:
        ret_dict[l[0]] = {}
        ret_dict[l[0]]["precision"] = float(l[1])
        ret_dict[l[0]]["recall"] = float(l[2])
        ret_dict[l[0]]["f1-score"] = float(l[3])
        ret_dict[l[0]]["support"] = float(l[4])

    return ret_dict

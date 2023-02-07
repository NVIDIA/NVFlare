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

import argparse
import copy
import json


def main():
    parser = argparse.ArgumentParser(description="merge two jsons together")
    parser.add_argument("--json_1", action="store", required=True, help="full path of json1")
    parser.add_argument("--json_2", action="store", help="full path of json2")
    parser.add_argument("--json_out", action="store", help="full path of json merged")
    args = parser.parse_args()

    json_1 = args.json_1
    json_2 = args.json_2
    json_out = args.json_out

    with open(json_1) as a:
        json_1_data = json.load(a)

    with open(json_2) as b:
        json_2_data = json.load(b)

    json_data = copy.deepcopy(json_1_data)
    json_data["training"].extend(json_2_data["training"])
    json_data["validation"].extend(json_2_data["validation"])
    json_data["testing"].extend(json_2_data["testing"])

    with open(json_out, "w") as f:
        json.dump(json_data, f, indent=4)

    return


if __name__ == "__main__":
    main()

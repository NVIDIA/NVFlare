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

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="./config_fed_client.json",
        help="config file in JSON format",
    )
    parser.add_argument(
        "--intersection_file",
        type=str,
        help="Intersection file with overlapping data indices",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    config["INTERSECTION_FILE"] = args.intersection_file

    with open(args.config_file, "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")

    print(f"Modified {args.config_file} to use INTERSECTION_FILE={config['INTERSECTION_FILE']}")


if __name__ == "__main__":
    main()

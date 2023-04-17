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
import os

from nvflare.fuel.f3.stats_pool import CsvRecordReader


def compute_time(file_name: str, pool_name: str):
    result = 0.0
    max_time = 0.0
    min_time = 1000
    count = 0

    print(f"Processing record file: {file_name}")
    reader = CsvRecordReader(file_name)
    for rec in reader:
        if rec.pool_name != pool_name:
            continue
        count += 1
        result += rec.value
        if max_time < rec.value:
            max_time = rec.value
        if min_time > rec.value:
            min_time = rec.value

    print(f"    Max={max_time};  Min={min_time};  Avg={result/count}; Count={count}; Total={result}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_dir", "-d", type=str, help="directory that contains stats record files", required=True)
    args = parser.parse_args()

    stats_dir = args.stats_dir
    files = os.listdir(stats_dir)
    if not files:
        print(f"no stats files in {stats_dir}")
        return -1

    total = 0.0
    for fn in files:
        if not fn.endswith(".csv"):
            continue
        t = compute_time(
            file_name=os.path.join(stats_dir, fn),
            pool_name="msg_travel",
        )
        total += t

    print(f"Total comm time: {total}")


if __name__ == "__main__":
    main()

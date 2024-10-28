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

"""
This tool can be used to compute the total time spent on communication for a job.

NOTE: if all processes (server and clients) are run on the same host, then the computed results are accurate.
If processes are run on different hosts, then these hosts must be synchronized with NTP (Network Time Protocol).

Before starting this tool, you must collect the stats_pool_records.csv files of all the processes into a folder. 
These files must all have the suffix of ".csv". For FL clients, these files are located in their workspaces.
For FL server, you need to download the job first (using admin console or flare api) and then find it in the downloaded 
workspace of the job.

Since these files have the same name in their workspaces, you must rename them when copying into the same folder.
You can simply use the client names for clients and "server" for server file.

Once you have all the csv files in the same folder, you can start this tool with the following args:

    -d: the directory that contains the csv files. Required.
    -o: the output file that will contain the result. Optional.

If the output file name is not specified, it will be default to "comm.txt".
The result is printed to the screen and written to the output file.

The output file will be placed into the same folder that contains the csv files. 
Do not name your output file with the suffix ".csv"!

"""


def _print(data: str, out_file):
    print(data)
    if out_file is not None:
        out_file.write(data + "\n")


def _compute_time(file_name: str, pool_name: str, out_file):
    result = 0.0
    max_time = 0.0
    min_time = 1000
    count = 0

    _print(f"Processing record file: {file_name}", out_file)
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

    _print(f"    Max={max_time};  Min={min_time};  Avg={result / count}; Count={count}; Total={result}", out_file)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_dir", "-d", type=str, help="directory that contains stats record files", required=True)
    parser.add_argument(
        "--out_file",
        "-o",
        type=str,
        help="directory that contains stats record files",
        required=False,
        default="comm.txt",
    )
    args = parser.parse_args()

    stats_dir = args.stats_dir
    files = os.listdir(stats_dir)
    if not files:
        print(f"No stats files in {stats_dir}")
        return -1

    out_file = None
    if args.out_file:
        out_file = open(os.path.join(stats_dir, args.out_file), "w")

    total = 0.0
    for fn in files:
        if not fn.endswith(".csv"):
            continue
        t = _compute_time(file_name=os.path.join(stats_dir, fn), pool_name="msg_travel", out_file=out_file)
        total += t

    _print(f"Total comm time: {total}", out_file)
    if out_file is not None:
        out_file.close()


if __name__ == "__main__":
    main()

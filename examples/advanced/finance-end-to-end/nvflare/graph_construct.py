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


import argparse
import os

import pandas as pd

# (1) import nvflare client API
import nvflare.client as flare

dataset_names = ["train", "test"]
datasets = {}


def main():
    print("\n pre-process starts \n ")
    args = define_parser()
    input_dir = args.input_dir
    output_dir = args.output_dir

    flare.init()
    site_name = flare.get_site_name()

    # receives global message from NVFlare
    etl_task = flare.receive()

    print("\n receive task \n ")
    edge_maps = edge_map_gen(input_dir, site_name)

    save_edge_map(output_dir, edge_maps, site_name)

    print("end task")

    # send message back the controller indicating end.
    etl_task.meta["status"] = "done"
    flare.send(etl_task)


def save_edge_map(output_dir, edge_maps, site_name):
    for name in edge_maps:
        site_dir = os.path.join(output_dir, site_name)
        os.makedirs(site_dir, exist_ok=True)

        edge_map_file_name = os.path.join(site_dir, f"{name}_edgemap.csv")
        print("save to = ", edge_map_file_name)
        # save to csv file without header and index
        edge_maps[name].to_csv(edge_map_file_name, header=False, index=False)


def edge_map_gen(input_dir, site_name):
    edge_maps = {}
    info_columns = ["Time", "Receiver_BIC", "UETR"]
    time_threshold = 6000
    for ds_name in dataset_names:

        file_name = os.path.join(input_dir, site_name, f"{ds_name}.csv")
        df = pd.read_csv(file_name)
        datasets[ds_name] = df

        # Find transaction pairs that are within the time threshold
        # First sort the table by 'Time'
        df = df.sort_values(by="Time")
        # Keep only the columns that are needed for the graph edge map
        df = df[info_columns]

        # Then for each row, find the next rows that is within the time threshold
        graph_edge_map = []
        for i in range(len(df)):
            # Find the next rows that is:
            # - within the time threshold
            # - has the same Receiver_BIC
            j = 1
            while i + j < len(df) and df["Time"].values[i + j] < df["Time"].values[i] + time_threshold:
                if df["Receiver_BIC"].values[i + j] == df["Receiver_BIC"].values[i]:
                    graph_edge_map.append([df["UETR"].values[i], df["UETR"].values[i + j]])
                j += 1

        print(
            f"Generated edge map for {ds_name}, in total {len(graph_edge_map)} valid edges for {len(df)} transactions"
        )

        edge_maps[ds_name] = pd.DataFrame(graph_edge_map)

    return edge_maps


def define_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        nargs="?",
        default="/tmp/dataset/credit_data",
        help="input directory where csv files for each site are expected, default to /tmp/dataset/credit_data",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        nargs="?",
        default="/tmp/dataset/credit_data",
        help="output directory, default to '/tmp/dataset/credit_data'",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()

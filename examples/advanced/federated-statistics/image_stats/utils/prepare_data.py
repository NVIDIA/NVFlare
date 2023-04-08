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
import glob
import json
import os
import random

SEED = 0


def create_datasets(root, subdirs, extension, shuffle, seed):
    random.seed(seed)

    data_lists = []
    for subdir in subdirs:
        search_string = os.path.join(root, "**", subdir, "images", "*" + extension)
        data_list = glob.glob(search_string, recursive=True)

        assert (
            len(data_list) > 0
        ), f"No images found using {search_string} for subdir '{subdir}' and extension '{extension}'!"

        if shuffle:
            random.shuffle(data_list)

        data_lists.append(data_list)

    return data_lists


def save_data_list(data, data_list_file, data_root, key="data"):
    data_list = []
    for d in data:
        data_list.append({"image": d.replace(data_root + os.path.sep, "")})

    os.makedirs(os.path.dirname(data_list_file), exist_ok=True)
    with open(data_list_file, "w") as f:
        json.dump({key: data_list}, f, indent=4)

    print(f"Saved {len(data_list)} entries at {data_list_file}")

def prepare_data(input_dir: str,
                 input_ext:str = ".png",
                 output_dir :str = "/tmp/nvflare/image_stats/data",
                 sub_dirs: str = "COVID,Lung_Opacity,Normal,Viral Pneumonia"):
    sub_dir_list = [sd for sd in sub_dirs.split(",")]

    data_lists = create_datasets(
        root=input_dir, subdirs=sub_dir_list, extension=input_ext, shuffle=True, seed=SEED
    )
    print(f"Created {len(data_lists)} data lists for {sub_dir_list}.")

    site_id = 1
    for subdir, data_list in zip(sub_dir_list, data_lists):
        save_data_list(
            data_list, os.path.join(output_dir, f"site-{site_id}_{subdir}.json"), data_root=input_dir
        )
        site_id += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Location of image files")
    parser.add_argument("--input_ext", type=str, default=".png", help="Search extension")
    parser.add_argument(
        "--output_dir", type=str, default="/tmp/nvflare/image_stats/data", help="Output location of data lists"
    )
    parser.add_argument(
        "--subdirs",
        type=str,
        default="COVID,Lung_Opacity,Normal,Viral Pneumonia",
        help="A list of subfolders to include.",
    )
    args = parser.parse_args()

    assert "," in args.subdirs, "Expecting a comma separated list of subdirs names"

    prepare_data(args.input_dir, args.input_ext, args.output_dir, args.subdirs)



if __name__ == "__main__":
    main()

# Copyright (c) 2021, NVIDIA CORPORATION.
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
import pathlib
import shutil
import subprocess


def clone_client(num_clients: int):
    current_path = os.getcwd()
    poc_folder = os.path.join(current_path, "poc")
    src_folder = os.path.join(poc_folder, "client")
    for index in range(1, num_clients + 1):
        dst_folder = os.path.join(poc_folder, f"site-{index}")
        shutil.copytree(src_folder, dst_folder)
        start_sh = open(os.path.join(dst_folder, "startup", "start.sh"), "rt")
        content = start_sh.read()
        start_sh.close()
        content = content.replace("NNN", f"{index}")
        with open(os.path.join(dst_folder, "startup", "start.sh"), "wt") as f:
            f.write(content)
    shutil.rmtree(src_folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_clients", type=int, default=1, help="number of client folders to create")

    args = parser.parse_args()

    file_path = pathlib.Path(__file__).parent.absolute()
    poc_zip_path = os.path.join(file_path, "..", "poc.zip")
    answer = input("This will delete poc folder in current directory and create a new one. Is it OK to proceed? (y/N) ")
    if answer.strip().upper() == "Y":
        shutil.rmtree(os.path.join(os.getcwd(), "poc"), ignore_errors=True)
        completed_process = subprocess.run(["unzip", "-q", poc_zip_path])
        returncode = completed_process.returncode
        if returncode != 0:
            print(f"Error during creating poc folder: {returncode=}")
            exit(returncode)
        clone_client(args.num_clients)
        print("Successfully creating poc folder.  Please read poc/Readme.rst for user guide.")


if __name__ == "__main__":
    main()

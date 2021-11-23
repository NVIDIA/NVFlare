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

from __future__ import absolute_import

import argparse
import os
import pathlib
import shutil
import sys

import yaml

from nvflare.fuel.utils.class_utils import instantiate_class
from nvflare.lighter.spec import Participant, Provisioner, Study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project_file", type=str, default="project.yml", help="file to describe FL project")
    parser.add_argument("-w", "--workspace", type=str, default="workspace", help="directory used by provision")
    parser.add_argument("-c", "--custom_folder", type=str, default=".", help="additional folder to load python codes")

    args = parser.parse_args()

    file_path = pathlib.Path(__file__).parent.absolute()
    current_path = os.getcwd()
    custom_folder_path = os.path.join(current_path, args.custom_folder)
    sys.path.append(custom_folder_path)
    print(f"Path list for python codes loading: {sys.path=}")

    # main project file
    project_file = args.project_file
    current_project_yml = os.path.join(current_path, "project.yml")
    if len(sys.argv) == 1 and not os.path.exists(current_project_yml):
        answer = input(
            f"No project.yml found in current folder.  Is it OK to generate one at {current_project_yml} for you? (y/N) "
        )
        if answer.strip().upper() == "Y":
            shutil.copyfile(os.path.join(file_path, "project.yml"), current_project_yml)
            print(f"{current_project_yml} was created.  Please edit it to fit your FL configuration.")
        exit(0)

    workspace = args.workspace
    workspace_full_path = os.path.join(current_path, workspace)

    project_full_path = os.path.join(current_path, project_file)
    print(f"Project yaml file: {project_full_path}.")

    project = yaml.load(open(project_full_path, "r"), Loader=yaml.Loader)
    api_version = project.get("api_version")
    if api_version not in [2]:
        raise ValueError(f"Incompatible API version found in {project_full_path}")

    study_name = project.get("name")
    study_description = project.get("description", "")
    participants = list()
    for p in project.get("participants"):
        participants.append(Participant(**p))
    study = Study(name=study_name, description=study_description, participants=participants)

    builders = list()
    for b in project.get("builders"):
        path = b.get("path")
        args = b.get("args")
        builders.append(instantiate_class(path, args))

    provisioner = Provisioner(workspace_full_path, builders)

    provisioner.provision(study)


if __name__ == "__main__":
    main()

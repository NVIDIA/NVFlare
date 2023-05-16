# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.fuel.utils.class_utils import instantiate_class
from nvflare.lighter.spec import Participant, Project, Provisioner
from nvflare.lighter.utils import load_yaml

adding_client_error_msg = """
name: $SITE-NAME
org: $ORGANIZATION_NAME
components:
    resource_manager:    # This id is reserved by system.  Do not change it.
        path: nvflare.app_common.resource_managers.gpu_resource_manager.GPUResourceManager
        args:
            num_of_gpus: 4,
            mem_per_gpu_in_GiB: 16
    resource_consumer:    # This id is reserved by system.  Do not change it.
        path: nvflare.app_common.resource_consumers.gpu_resource_consumer.GPUResourceConsumer
        args:
"""

adding_user_error_msg = """
name: $USER_EMAIL_ADDRESS
org: $ORGANIZATION_NAME
role: $ROLE
"""


def define_provision_parser(parser):
    parser.add_argument("-p", "--project_file", type=str, default="project.yml", help="file to describe FL project")
    parser.add_argument("-w", "--workspace", type=str, default="workspace", help="directory used by provision")
    parser.add_argument("-c", "--custom_folder", type=str, default=".", help="additional folder to load python codes")
    parser.add_argument("--add_user", type=str, default="", help="yaml file for added user")
    parser.add_argument("--add_client", type=str, default="", help="yaml file for added client")


def has_no_arguments() -> bool:
    last_item = sys.argv[-1]
    return last_item.endswith("provision") or last_item.endswith("provision.py")


def handle_provision(args):
    file_path = pathlib.Path(__file__).parent.absolute()
    current_path = os.getcwd()
    custom_folder_path = os.path.join(current_path, args.custom_folder)
    sys.path.append(custom_folder_path)

    # main project file
    project_file = args.project_file
    current_project_yml = os.path.join(current_path, "project.yml")

    if has_no_arguments() and not os.path.exists(current_project_yml):
        files = {"1": "ha_project.yml", "2": "dummy_project.yml", "3": None}
        print("No project.yml found in current folder.\nThere are two types of templates for project.yml.")
        print(
            "1) project.yml for HA mode\n2) project.yml for non-HA mode\n3) Don't generate project.yml.  Exit this program."
        )
        answer = input(f"Which type of project.yml should be generated at {current_project_yml} for you? (1/2/3) ")
        answer = answer.strip()
        src_project = files.get(answer, None)
        if src_project:
            shutil.copyfile(os.path.join(file_path, src_project), current_project_yml)
            print(
                f"{current_project_yml} was created.  Please edit it to fit your FL configuration. "
                + "Once done please run nvflare provision command again with newly edited project.yml file"
            )

        else:
            print(f"{answer} was selected.  No project.yml was created.")
        exit(0)

    workspace = args.workspace
    workspace_full_path = os.path.join(current_path, workspace)

    project_full_path = os.path.join(current_path, project_file)
    print(f"Project yaml file: {project_full_path}.")

    project_dict = load_yaml(project_full_path)
    api_version = project_dict.get("api_version")
    if api_version not in [3]:
        raise ValueError(f"API version expected 3 but found {api_version}")

    project_name = project_dict.get("name")
    project_description = project_dict.get("description", "")
    participants = list()
    for p in project_dict.get("participants"):
        participants.append(Participant(**p))
    if args.add_user:
        try:
            extra = load_yaml(os.path.join(current_path, args.add_user))
            extra.update({"type": "admin"})
            participants.append(Participant(**extra))
        except Exception:
            print("** Error during adding user **")
            print("The yaml file format is")
            print(adding_user_error_msg)
            exit(0)
    if args.add_client:
        try:
            extra = load_yaml(os.path.join(current_path, args.add_client))
            extra.update({"type": "client"})
            participants.append(Participant(**extra))
        except Exception as e:
            print("** Error during adding client **")
            print("The yaml file format is")
            print(adding_client_error_msg)
            exit(0)

    project = Project(name=project_name, description=project_description, participants=participants)

    n_servers = len(project.get_participants_by_type("server", first_only=False))
    if n_servers > 2:
        print(
            f"Configuration error: Expect 2 or 1 server to be provisioned.  {project_full_path} contains {n_servers} servers."
        )
        return

    builders = list()
    for b in project_dict.get("builders"):
        path = b.get("path")
        args = b.get("args")
        builders.append(instantiate_class(path, args))

    provisioner = Provisioner(workspace_full_path, builders)

    provisioner.provision(project)


def main():
    print("*****************************************************************************")
    print("** provision command is deprecated, please use 'nvflare provision' instead **")
    print("*****************************************************************************")

    parser = argparse.ArgumentParser()
    define_provision_parser(parser)
    args = parser.parse_args()
    handle_provision(args)


if __name__ == "__main__":
    main()

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
from typing import Optional

from nvflare.lighter.constants import PropKey
from nvflare.lighter.entity import participant_from_dict
from nvflare.lighter.prov_utils import prepare_builders, prepare_packager
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.spec import Project
from nvflare.lighter.tree_prov import hierachical_provision
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


_provision_parser = None


def define_provision_parser(parser):
    global _provision_parser
    _provision_parser = parser
    # Action flags — mutually exclusive but no longer required; default is -g behavior
    parser.add_argument("-p", "--project_file", type=str, default=None, help="file to describe FL project")
    parser.add_argument(
        "-g",
        "--generate",
        action="store_true",
        help="generate a sample project.yml and exit (default when no flag given)",
    )
    parser.add_argument("-e", "--gen_edge", action="store_true", help="generate a sample edge project.yml and exit")

    # Optional arguments
    parser.add_argument("-w", "--workspace", type=str, default="workspace", help="directory used by provision")
    parser.add_argument("-c", "--custom_folder", type=str, default=".", help="additional folder to load python codes")
    parser.add_argument("--add_user", type=str, default="", help="yaml file for added user")
    parser.add_argument("--add_client", type=str, default="", help="yaml file for added client")
    parser.add_argument("-s", "--gen_scripts", action="store_true", help="generate test scripts like start_all.sh")
    parser.add_argument("--force", action="store_true", help="skip Y/N confirmation prompts")
    parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")


def copy_project(project: str, dest: str):
    file_path = pathlib.Path(__file__).parent.absolute()
    dummy_project = os.path.join(file_path, project)
    shutil.copyfile(dummy_project, dest)
    rel_path = os.path.relpath(dest)
    from nvflare.tool.cli_output import print_human

    print_human(
        f"{dest} was generated.  Please edit it to fit your NVFlare configuration. "
        + f"Once done please run 'nvflare provision -p {rel_path}' to perform the provisioning"
    )


def handle_provision(args):
    from nvflare.tool.cli_output import output_ok
    from nvflare.tool.cli_schema import handle_schema_flag
    from nvflare.tool.install_skills import install_skills

    handle_schema_flag(
        _provision_parser,
        "nvflare provision",
        ["nvflare provision -p project.yml", "nvflare provision -g"],
        sys.argv[1:],
    )
    current_path = os.getcwd()
    custom_folder_path = os.path.join(current_path, args.custom_folder)
    sys.path.append(custom_folder_path)

    current_project_yml = os.path.join(current_path, "project.yml")

    # Default when no project_file and no -g: generate sample project.yml (pre-2.7.0 behavior)
    if args.gen_edge:
        copy_project("edge_project.yml", current_project_yml)
        output_ok({"workspace": current_path, "packages": [], "project_yml": current_project_yml})
        try:
            install_skills()
        except Exception:
            pass
        return

    if not args.project_file or args.generate:
        copy_project("dummy_project.yml", current_project_yml)
        output_ok({"workspace": current_path, "packages": [], "project_yml": current_project_yml})
        try:
            install_skills()
        except Exception:
            pass
        return

    # main project file
    project_file = args.project_file

    workspace = args.workspace
    workspace_full_path = os.path.join(current_path, workspace)

    project_full_path = os.path.join(current_path, project_file)
    from nvflare.tool.cli_output import print_human

    print_human(f"Project yaml file: {project_full_path}.")

    add_user_full_path = os.path.join(current_path, args.add_user) if args.add_user else None
    add_client_full_path = os.path.join(current_path, args.add_client) if args.add_client else None

    provision(args, project_full_path, workspace_full_path, add_user_full_path, add_client_full_path)

    # Collect packages from workspace
    packages = []
    if os.path.isdir(workspace_full_path):
        for item in os.listdir(workspace_full_path):
            item_path = os.path.join(workspace_full_path, item)
            if os.path.isdir(item_path):
                packages.append(item)

    output_ok({"workspace": workspace_full_path, "packages": packages})
    from nvflare.tool.cli_output import print_human

    print_human(f"\nProvisioning complete. Packages written to: {workspace_full_path}")
    if packages:
        print_human(f"  Packages: {', '.join(packages)}")
        print_human("  Verify each package with: nvflare preflight -p <package_path>")
    print_human("  Distribute packages to each participant and run their start.sh")
    try:
        install_skills()
    except Exception:
        pass


def gen_default_project_config(src_project_name, dest_project_file):
    file_path = pathlib.Path(__file__).parent.absolute()
    shutil.copyfile(os.path.join(file_path, src_project_name), dest_project_file)


def provision_for_edge(params, project_dict):
    project_name = project_dict.get(PropKey.NAME)
    project_description = project_dict.get(PropKey.DESCRIPTION, "")
    project = Project(name=project_name, description=project_description, props=project_dict)

    participants = project_dict.get("participants")
    admins = [participant_from_dict(p) for p in participants if p.get("type") == "admin"]
    builders = prepare_builders(project_dict)
    hierachical_provision(params, project, builders, admins)


def provision(
    args,
    project_full_path: str,
    workspace_full_path: str,
    add_user_full_path: Optional[str] = None,
    add_client_full_path: Optional[str] = None,
):
    project_dict = load_yaml(project_full_path)
    project_dict["gen_scripts"] = args.gen_scripts
    edge_params = project_dict.get("edge")
    if edge_params:
        try:
            provision_for_edge(edge_params, project_dict)
        except Exception as e:
            raise Exception(f"Provisioning failed in edge mode: {e}")
        return

    project = prepare_project(project_dict, add_user_full_path, add_client_full_path)
    builders = prepare_builders(project_dict)
    packager = prepare_packager(project_dict)
    provisioner = Provisioner(workspace_full_path, builders, packager)
    provisioner.provision(project)


def prepare_project(project_dict, add_user_file_path=None, add_client_file_path=None):
    api_version = project_dict.get(PropKey.API_VERSION)
    if api_version not in [3]:
        raise ValueError(f"API version expected 3 but found {api_version}")
    project_name = project_dict.get(PropKey.NAME)
    if len(project_name) > 63:
        from nvflare.tool.cli_output import print_human

        print_human(f"Project name {project_name} is longer than 63.  Will truncate it to {project_name[:63]}.")
        project_name = project_name[:63]
        project_dict[PropKey.NAME] = project_name
    project_description = project_dict.get(PropKey.DESCRIPTION, "")
    project = Project(name=project_name, description=project_description, props=project_dict)
    participant_defs = project_dict.get("participants")

    if add_user_file_path:
        add_extra_users(add_user_file_path, participant_defs)

    if add_client_file_path:
        add_extra_clients(add_client_file_path, participant_defs)

    for p in participant_defs:
        project.add_participant(participant_from_dict(p))
    return project


def add_extra_clients(add_client_file_path, participant_defs):
    try:
        extra = load_yaml(add_client_file_path)
        extra.update({"type": "client"})
        participant_defs.append(extra)
    except Exception:
        from nvflare.tool.cli_output import output_error, print_human

        print_human("** Error during adding client **")
        print_human("The yaml file format is")
        print_human(adding_client_error_msg)
        output_error("INVALID_ARGS", exit_code=4, detail="invalid client yaml format")


def add_extra_users(add_user_file_path, participant_defs):
    try:
        extra = load_yaml(add_user_file_path)
        extra.update({"type": "admin"})
        participant_defs.append(extra)
    except Exception:
        from nvflare.tool.cli_output import output_error, print_human

        print_human("** Error during adding user **")
        print_human("The yaml file format is")
        print_human(adding_user_error_msg)
        output_error("INVALID_ARGS", exit_code=4, detail="invalid user yaml format")


def main():
    from nvflare.tool.cli_output import print_human

    print_human("*****************************************************************************")
    print_human("** provision command is deprecated, please use 'nvflare provision' instead **")
    print_human("*****************************************************************************")

    parser = argparse.ArgumentParser()
    define_provision_parser(parser)
    args = parser.parse_args()
    handle_provision(args)


if __name__ == "__main__":
    main()

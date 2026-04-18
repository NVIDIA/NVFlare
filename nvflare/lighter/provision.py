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

from nvflare.apis.utils.format_check import name_check
from nvflare.lighter.constants import AdminRole, CtxKey, ParticipantType, PropKey
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


def _normalize_and_validate_studies(project_dict: dict, participant_defs: list, api_version: int) -> dict:
    studies = project_dict.get("studies")
    if studies is None:
        return {}

    if api_version != 4:
        raise ValueError("studies: requires api_version: 4")

    if not isinstance(studies, dict):
        raise ValueError(f"studies must be a mapping but got {type(studies)}")

    client_names = {p.get("name") for p in participant_defs if p.get("type") == ParticipantType.CLIENT}
    admin_names = {p.get("name") for p in participant_defs if p.get("type") == ParticipantType.ADMIN}

    normalized = {}
    valid_roles = {AdminRole.PROJECT_ADMIN, AdminRole.ORG_ADMIN, AdminRole.LEAD, AdminRole.MEMBER}
    for study_name, study_def in studies.items():
        if study_name == "default":
            raise ValueError("study name 'default' is reserved")

        err, reason = name_check(study_name, "study")
        if err:
            raise ValueError(f"invalid study name '{study_name}': {reason}")

        if study_def is None:
            normalized[study_name] = {}
            continue

        if not isinstance(study_def, dict):
            raise ValueError(f"study '{study_name}' must be a mapping")

        sites = study_def.get("sites", [])
        admins = study_def.get("admins", {})
        if sites is None:
            sites = []
        if admins is None:
            admins = {}

        if not isinstance(sites, list):
            raise ValueError(f"study '{study_name}' sites must be a list")
        if not isinstance(admins, dict):
            raise ValueError(f"study '{study_name}' admins must be a mapping")

        for site in sites:
            if site not in client_names:
                raise ValueError(f"study '{study_name}' references unknown client '{site}'")

        for admin_name, role in admins.items():
            if admin_name not in admin_names:
                raise ValueError(f"study '{study_name}' references unknown admin '{admin_name}'")
            if role not in valid_roles:
                raise ValueError(f"study '{study_name}' assigns unknown role '{role}' to '{admin_name}'")

        normalized[study_name] = dict(study_def)

    return normalized


def _project_generation_result(workspace: str, project_yml: str):
    rel_path = os.path.basename(project_yml)
    return {
        "workspace": workspace,
        "packages": [],
        "project_yml": project_yml,
        "message": "Sample project file generated.",
        "next_step": "Edit the project file, then run provisioning.",
        "suggested_command": f"nvflare provision -p {rel_path}",
    }


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
    from nvflare.tool.cli_output import is_json_mode, print_human

    if not is_json_mode():
        print_human(
            f"{dest} was generated.  Please edit it to fit your NVFlare configuration. "
            + f"Once done please run 'nvflare provision -p {rel_path}' to perform the provisioning"
        )


def handle_provision(args):
    from nvflare.tool.cli_output import output_error, output_ok
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

    if args.generate and args.project_file:
        output_error("INVALID_ARGS", exit_code=4, detail="cannot use -p/--project_file together with -g/--generate")
        raise SystemExit(4)

    # Default when no project_file and no -g: generate sample project.yml (pre-2.7.0 behavior)
    if args.gen_edge:
        copy_project("edge_project.yml", current_project_yml)
        output_ok(_project_generation_result(current_path, current_project_yml))
        try:
            install_skills()
        except Exception:
            pass
        return

    if not args.project_file or args.generate:
        copy_project("dummy_project.yml", current_project_yml)
        output_ok(_project_generation_result(current_path, current_project_yml))
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
    if not os.path.isfile(project_full_path):
        output_error("INVALID_ARGS", exit_code=4, detail=f"project file does not exist: {project_full_path}")
        raise SystemExit(4)
    from nvflare.tool.cli_output import is_json_mode, print_human

    try:
        project_dict = load_yaml(project_full_path)
    except Exception as e:
        output_error(
            "INVALID_ARGS",
            exit_code=4,
            detail=f"project file is empty or not a valid YAML mapping: {project_full_path}: {e}",
        )
        raise SystemExit(4)
    if not project_dict or not isinstance(project_dict, dict):
        output_error(
            "INVALID_ARGS",
            exit_code=4,
            detail=f"project file is empty or not a valid YAML mapping: {project_full_path}",
        )
        raise SystemExit(4)
    project_name = project_dict.get(PropKey.NAME)
    if not project_name:
        output_error("INVALID_ARGS", exit_code=4, detail="missing project name")
        raise SystemExit(4)
    project_workspace = os.path.join(workspace_full_path, project_name)
    if os.path.isdir(project_workspace) and os.listdir(project_workspace):
        from nvflare.tool.cli_output import prompt_yn

        if not args.force:
            if not sys.stdin.isatty():
                output_error(
                    "INVALID_ARGS",
                    exit_code=4,
                    detail="workspace exists; use --force to continue in non-interactive mode",
                )
                raise SystemExit(4)
            if not prompt_yn(
                f"Provision workspace already exists for project '{project_name}' at '{project_workspace}'. Continue?"
            ):
                return

    if not is_json_mode():
        print_human(f"Project yaml file: {project_full_path}.")

    add_user_full_path = os.path.join(current_path, args.add_user) if args.add_user else None
    add_client_full_path = os.path.join(current_path, args.add_client) if args.add_client else None
    if add_user_full_path and not os.path.isfile(add_user_full_path):
        output_error("INVALID_ARGS", exit_code=4, detail=f"add_user file does not exist: {add_user_full_path}")
        raise SystemExit(4)
    if add_client_full_path and not os.path.isfile(add_client_full_path):
        output_error("INVALID_ARGS", exit_code=4, detail=f"add_client file does not exist: {add_client_full_path}")
        raise SystemExit(4)

    ctx = provision(args, project_full_path, workspace_full_path, add_user_full_path, add_client_full_path)

    if isinstance(ctx, dict) and ctx.get(CtxKey.BUILD_ERROR):
        diagnostic_lines = []
        errors = ctx.get(CtxKey.ERRORS, [])
        warnings = ctx.get(CtxKey.WARNINGS, [])
        if errors:
            diagnostic_lines.append("Errors:")
            diagnostic_lines.extend(f"- {msg}" for msg in errors)
        if warnings:
            diagnostic_lines.append("Warnings:")
            diagnostic_lines.extend(f"- {msg}" for msg in warnings)
        detail = "\n".join(diagnostic_lines) if diagnostic_lines else "Provisioning failed during kit assembly."
        output_error("INTERNAL_ERROR", exit_code=5, detail=detail)
        raise SystemExit(5)

    # Collect packages from workspace
    packages = []
    if os.path.isdir(workspace_full_path):
        for item in os.listdir(workspace_full_path):
            item_path = os.path.join(workspace_full_path, item)
            if os.path.isdir(item_path):
                packages.append(item)

    output_ok({"workspace": workspace_full_path, "packages": packages})

    if not is_json_mode():
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
    if not project_name:
        raise ValueError("missing project name")
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
            from nvflare.tool.cli_output import output_error

            output_error("INTERNAL_ERROR", exit_code=5, detail=f"Provisioning failed in edge mode: {e}")
            raise SystemExit(5)
        return None

    project = prepare_project(project_dict, add_user_full_path, add_client_full_path)
    builders = prepare_builders(project_dict)
    packager = prepare_packager(project_dict)
    provisioner = Provisioner(workspace_full_path, builders, packager)
    return provisioner.provision(project)


def prepare_project(project_dict, add_user_file_path=None, add_client_file_path=None):
    api_version = project_dict.get(PropKey.API_VERSION)
    if api_version not in [3, 4]:
        raise ValueError(f"API version expected 3 or 4 but found {api_version}")
    project_name = project_dict.get(PropKey.NAME)
    if not project_name:
        raise ValueError("missing project name")
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

    project_dict["studies"] = _normalize_and_validate_studies(project_dict, participant_defs, api_version)

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
        raise SystemExit(4)


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
        raise SystemExit(4)


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

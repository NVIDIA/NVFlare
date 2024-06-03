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

import os
import shutil
import traceback
from distutils.dir_util import copy_tree
from tempfile import mkdtemp
from typing import List, Optional, Tuple

from pyhocon import ConfigFactory as CF
from pyhocon import ConfigTree

from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException
from nvflare.fuel.flare_api.flare_api import new_secure_session
from nvflare.fuel.utils.config import ConfigFormat
from nvflare.fuel.utils.config_factory import ConfigFactory
from nvflare.tool.job.config.configer import (
    build_config_file_indices,
    filter_indices,
    get_root_index,
    merge_configs_from_cli,
)
from nvflare.tool.job.job_client_const import (
    CONFIG_FED_CLIENT_CONF,
    CONFIG_FED_SERVER_CONF,
    CONFIG_FILE_BASE_NAME_WO_EXTS,
    DEFAULT_APP_NAME,
    JOB_CONFIG_COMP_NAME,
    JOB_CONFIG_FILE_NAME,
    JOB_CONFIG_VAR_NAME,
    JOB_CONFIG_VAR_VALUE,
    JOB_INFO_CONF,
    JOB_INFO_CONTROLLER_TYPE,
    JOB_INFO_CONTROLLER_TYPE_KEY,
    JOB_INFO_DESC,
    JOB_INFO_DESC_KEY,
    JOB_INFO_EXECUTION_API_TYPE,
    JOB_INFO_EXECUTION_API_TYPE_KEY,
    JOB_INFO_KEYS,
    JOB_INFO_MD,
    JOB_META_BASE_NAME,
    META_APP_NAME,
    TEMPLATES_KEY,
)
from nvflare.utils.cli_utils import (
    create_job_template_config,
    find_job_templates_location,
    get_curr_dir,
    get_hidden_config,
    get_startup_kit_dir,
    save_config,
)

CMD_LIST_TEMPLATES = "list_templates"
CMD_SHOW_VARIABLES = "show_variables"
CMD_CREATE_JOB = "create"
CMD_SUBMIT_JOB = "submit"


def find_filename_basename(f: str):
    basename = os.path.basename(f)
    if "." in basename:
        return os.path.splitext(basename)[0]
    else:
        return basename


def build_job_template_indices(job_templates_dir: str) -> ConfigTree:
    conf = CF.parse_string("{ templates = {} }")
    config_file_base_names = CONFIG_FILE_BASE_NAME_WO_EXTS
    template_conf = conf.get(TEMPLATES_KEY)
    keys = JOB_INFO_KEYS

    for f in os.listdir(job_templates_dir):
        template_path = os.path.join(job_templates_dir, f)
        if os.path.isdir(template_path):
            for _, _, files in os.walk(template_path):
                config_files = [f for f in files if find_filename_basename(f) in config_file_base_names]
                if len(config_files) > 0:
                    info_conf = get_template_info_config(template_path)
                    for key in keys:
                        value = info_conf.get(key, "NA") if info_conf else "NA"
                        template_name = os.path.basename(f)
                        template_conf.put(f"{template_name}.{key}", value)

    return conf


def get_template_info_config(template_dir):
    info_conf_path = os.path.join(template_dir, JOB_INFO_CONF)
    return CF.parse_file(info_conf_path) if os.path.isfile(info_conf_path) else None


def get_app_dirs_from_template(template_dir):
    app_dirs = []
    for root, dirs, files in os.walk(template_dir):
        if root != template_dir and (CONFIG_FED_SERVER_CONF in files or CONFIG_FED_CLIENT_CONF in files):
            app_dirs.append(root)

    return app_dirs


def get_app_dirs_from_job_folder(job_folder):
    app_dirs = []
    for root, dirs, files in os.walk(job_folder):
        if root != job_folder and (root.endswith("config") or root.endswith("custom")):
            dir_name = os.path.dirname(os.path.relpath(root, job_folder))
            if dir_name:
                app_dirs.append(dir_name)

    return app_dirs


def create_job(cmd_args):
    try:
        template_src = get_src_template(cmd_args)
        if not template_src:
            template_src = get_src_template_by_name(cmd_args)
        app_dirs = get_app_dirs_from_template(str(template_src).strip())
        app_names = [os.path.basename(f) for f in app_dirs]
        app_names = app_names if app_names else [DEFAULT_APP_NAME]
        job_folder = cmd_args.job_folder
        prepare_job_folder(cmd_args)
        app_custom_dirs = prepare_app_dirs(job_folder, app_names)
        prepare_app_scripts(job_folder, app_custom_dirs, cmd_args)
        config_dirs = get_config_dirs(job_folder, app_names)

        fmt, real_config_path = ConfigFactory.search_config_format(CONFIG_FED_CLIENT_CONF, config_dirs)
        if real_config_path and not cmd_args.force:
            print(
                f"""\nwarning: configuration files:\n
                    {"config_fed_server.[json|conf|yml]"} already exists.
                \nNot generating the config files. If you would like to overwrite, use -force option"""
            )
            return

        template_srcs = {}
        if not app_dirs:
            template_srcs[DEFAULT_APP_NAME] = template_src
        else:
            for app_dir in app_dirs:
                app_name = os.path.basename(app_dir)
                template_srcs[app_name] = app_dir

        for app_name in template_srcs:
            src = template_srcs[app_name]
            app_config_dir = get_config_dir(job_folder, app_name)
            copy_tree(src=src, dst=app_config_dir)
            remove_extra_files(app_config_dir)
        prepare_meta_config(cmd_args, template_src, app_names)
        app_variable_values = prepare_job_config(cmd_args, app_names)
        display_template_variables(job_folder, app_variable_values)

    except ValueError as e:
        print(f"\nUnable to handle command: {CMD_CREATE_JOB} due to: {e} \n")
        if cmd_args.debug:
            print(traceback.format_exc())
        sub_cmd_parser = job_sub_cmd_parser[CMD_CREATE_JOB]
        if sub_cmd_parser:
            sub_cmd_parser.print_help()


def get_src_template_by_name(cmd_args):
    job_templates_dir = find_job_templates_location()
    template_index_conf = build_job_template_indices(job_templates_dir)
    target_template_name = cmd_args.template
    check_template_exists(target_template_name, template_index_conf)
    template_src = os.path.join(job_templates_dir, target_template_name)
    return template_src


def get_src_template(cmd_args) -> Optional[str]:
    target_template = os.path.abspath(cmd_args.template)
    if os.path.isdir(target_template):
        info_file = os.path.join(target_template, JOB_INFO_CONF)
        if os.path.isfile(info_file):
            return target_template

    return None


def remove_pycache_files(custom_dir):
    for root, dirs, files in os.walk(custom_dir):
        # remove pycache and pyc files
        for d in dirs:
            if d == "__pycache__" or d.endswith(".pyc"):
                shutil.rmtree(os.path.join(root, d))


def remove_extra_files(config_dir):
    extra_file = [JOB_INFO_MD, JOB_INFO_CONF, "__init__.py", "__pycache__"]
    for ef in extra_file:
        file_path = os.path.join(config_dir, ef)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def show_variables(cmd_args):
    try:
        if not os.path.isdir(cmd_args.job_folder):
            raise ValueError("required job folder is not specified.")

        app_dirs = get_app_dirs_from_job_folder(cmd_args.job_folder)
        app_names = [os.path.basename(f) for f in app_dirs]
        app_names = app_names if app_names else [DEFAULT_APP_NAME]
        indices = build_config_file_indices(cmd_args.job_folder, app_names)
        variable_values = filter_indices(app_indices_configs=indices)
        display_template_variables(cmd_args.job_folder, variable_values)

    except ValueError as e:
        print(f"\nUnable to handle command: {CMD_SHOW_VARIABLES} due to: {e} \n")
        if cmd_args.debug:
            print(traceback.format_exc())
        sub_cmd_parser = job_sub_cmd_parser[CMD_SHOW_VARIABLES]
        if sub_cmd_parser:
            sub_cmd_parser.print_help()


def check_template_exists(target_template_name, template_index_conf):
    targets = [os.path.basename(key) for key in template_index_conf.get("templates").keys()]
    found = target_template_name in targets

    if not found:
        raise ValueError(
            f"Invalid template name {target_template_name}, "
            f"please check the available templates using nvflare job list_templates"
        )


def display_template_variables(job_folder, app_variable_values):
    print("\nThe following are the variables you can change in the template\n")
    total_length = 135
    left_margin = 1
    print("-" * total_length)
    job_folder_header = fix_length_format(f"job folder: {job_folder}", total_length)
    print(" " * total_length)
    print(" " * left_margin, job_folder_header)
    print(" " * total_length)
    print("-" * total_length)
    file_name_fix_length = 30
    var_name_fix_length = 30
    var_value_fix_length = 35
    var_comp_fix_length = 35
    file_name = fix_length_format(JOB_CONFIG_FILE_NAME, file_name_fix_length)
    var_name = fix_length_format(JOB_CONFIG_VAR_NAME, var_name_fix_length)
    var_value = fix_length_format(JOB_CONFIG_VAR_VALUE, var_value_fix_length)
    var_comp = fix_length_format(JOB_CONFIG_COMP_NAME, var_comp_fix_length)
    print(" " * left_margin, file_name, var_name, var_value, var_comp)
    print("-" * total_length)
    for app_name, variable_values in app_variable_values.items():
        if app_name != DEFAULT_APP_NAME and app_name != META_APP_NAME:
            app_header = fix_length_format(f"app: {app_name}", total_length)
            print(" " * left_margin, app_header)
            print(" " * total_length)

        for file in sorted(variable_values.keys()):
            indices = variable_values.get(file)
            file_name = os.path.basename(file)
            file_name = fix_length_format(file_name, file_name_fix_length)
            key_indices = indices

            for index in sorted(key_indices.keys()):
                key_index = key_indices[index]
                var_name = fix_length_format(index, var_name_fix_length)
                var_value = fix_length_format(str(key_index.value), var_value_fix_length)
                var_comp = " " if key_index.component_name is None else key_index.component_name
                var_comp = fix_length_format(var_comp, var_comp_fix_length)
                print(" " * left_margin, file_name, var_name, var_value, var_comp)

            print("")
    print("-" * total_length)


def list_templates(cmd_args):
    try:
        job_templates_dir = find_job_templates_location(cmd_args.job_templates_dir)
        job_templates_dir = os.path.abspath(job_templates_dir)
        template_index_conf = build_job_template_indices(job_templates_dir)
        display_available_templates(template_index_conf)

        if job_templates_dir:
            update_job_templates_dir(job_templates_dir)

    except ValueError as e:
        print(f"\nUnable to handle command: {CMD_LIST_TEMPLATES} due to: {e} \n")
        if cmd_args.debug:
            print(traceback.format_exc())
        sub_cmd_parser = job_sub_cmd_parser[CMD_LIST_TEMPLATES]
        if sub_cmd_parser:
            sub_cmd_parser.print_help()


def update_job_templates_dir(job_templates_dir: str):
    config_file_path, nvflare_config = get_hidden_config()
    config = create_job_template_config(nvflare_config, job_templates_dir)
    save_config(config, config_file_path)


def display_available_templates(template_index_conf):
    print("\nThe following job templates are available: \n")
    template_registry = template_index_conf.get("templates")
    total_length = 120
    left_margin = 1
    print("-" * total_length)
    name_fix_length = 20
    description_fix_length = 60
    controller_type_fix_length = 17
    execution_api_type_fix_length = 23
    name = fix_length_format("name", name_fix_length)
    description = fix_length_format(JOB_INFO_DESC, description_fix_length)
    execution_api_type = fix_length_format(JOB_INFO_EXECUTION_API_TYPE, execution_api_type_fix_length)
    controller_type = fix_length_format(JOB_INFO_CONTROLLER_TYPE, controller_type_fix_length)
    print(" " * left_margin, name, description, controller_type, execution_api_type)
    print("-" * total_length)
    for file_path in sorted(template_registry.keys()):
        name = os.path.basename(file_path)
        template_info = template_registry.get(file_path, None)
        if not template_info:
            template_info = template_registry.get(name)
        name = fix_length_format(name, name_fix_length)
        description = fix_length_format(template_info.get(JOB_INFO_DESC_KEY), description_fix_length)
        execution_api_type = fix_length_format(
            template_info.get(JOB_INFO_EXECUTION_API_TYPE_KEY), execution_api_type_fix_length
        )
        controller_type = fix_length_format(template_info.get(JOB_INFO_CONTROLLER_TYPE_KEY), controller_type_fix_length)
        print(" " * left_margin, name, description, controller_type, execution_api_type)
    print("-" * total_length)


def fix_length_format(name: str, name_fix_length: int):
    return f"{name[:name_fix_length]:{name_fix_length}}"


def submit_job(cmd_args):
    temp_job_dir = None
    try:
        if not os.path.isdir(cmd_args.job_folder):
            raise ValueError(f"invalid job folder: {cmd_args.job_folder}")

        temp_job_dir = mkdtemp()
        copy_tree(cmd_args.job_folder, temp_job_dir)

        app_dirs = get_app_dirs_from_job_folder(cmd_args.job_folder)
        app_names = [os.path.basename(f) for f in app_dirs]
        app_names = app_names if app_names else [DEFAULT_APP_NAME]

        prepare_job_config(cmd_args, app_names, temp_job_dir)
        admin_username, admin_user_dir = find_admin_user_and_dir()
        internal_submit_job(admin_user_dir, admin_username, temp_job_dir)

    except ValueError as e:
        print(f"\nUnable to handle command: {CMD_SUBMIT_JOB} due to: {e} \n")
        if cmd_args.debug:
            print(traceback.format_exc())
        sub_cmd_parser = job_sub_cmd_parser[CMD_SUBMIT_JOB]
        if sub_cmd_parser:
            sub_cmd_parser.print_help()
    finally:
        if temp_job_dir:
            if cmd_args.debug:
                print(f"in debug mode, job configurations can be examined in temp job directory '{temp_job_dir}'")
            else:
                shutil.rmtree(temp_job_dir)


def find_admin_user_and_dir() -> Tuple[str, str]:
    startup_kit_dir = get_startup_kit_dir()
    fed_admin_config = ConfigFactory.load_config("fed_admin.json", [startup_kit_dir])

    if fed_admin_config:
        admin_user_dir = os.path.dirname(os.path.dirname(fed_admin_config.file_path))
        config_dict = fed_admin_config.to_dict()
        admin_username = config_dict["admin"].get("username", None)
    else:
        raise ValueError(f"Unable to locate fed_admin configuration from startup kid location {startup_kit_dir}")

    return admin_username, admin_user_dir


def internal_submit_job(admin_user_dir, username, temp_job_dir):
    print("trying to connect to the server")
    sess = new_secure_session(username=username, startup_kit_location=admin_user_dir)
    job_id = sess.submit_job(temp_job_dir)
    print(f"job: '{job_id} was submitted")


job_sub_cmd_handlers = {
    CMD_CREATE_JOB: create_job,
    CMD_SUBMIT_JOB: submit_job,
    CMD_LIST_TEMPLATES: list_templates,
    CMD_SHOW_VARIABLES: show_variables,
}

job_sub_cmd_parser = {
    CMD_CREATE_JOB: None,
    CMD_SUBMIT_JOB: None,
    CMD_LIST_TEMPLATES: None,
    CMD_SHOW_VARIABLES: None,
}


def handle_job_cli_cmd(cmd_args):
    job_cmd_handler = job_sub_cmd_handlers.get(cmd_args.job_sub_cmd, None)
    if job_cmd_handler:
        job_cmd_handler(cmd_args)
    else:
        raise CLIUnknownCmdException("\n invalid command. \n")


def def_job_cli_parser(sub_cmd):
    cmd = "job"
    parser = sub_cmd.add_parser(cmd)
    job_subparser = parser.add_subparsers(title="job", dest="job_sub_cmd", help="job subcommand")
    define_list_templates_parser(job_subparser)
    define_create_job_parser(job_subparser)
    define_submit_job_parser(job_subparser)
    define_variables_parser(job_subparser)

    return {cmd: parser}


def define_submit_job_parser(job_subparser):
    submit_parser = job_subparser.add_parser("submit", help="submit job")
    submit_parser.add_argument(
        "-j",
        "--job_folder",
        type=str,
        nargs="?",
        default=os.path.join(get_curr_dir(), "current_job"),
        help="job_folder path, default to ./current_job directory",
    )
    submit_parser.add_argument(
        "-f",
        "--config_file",
        type=str,
        action="append",
        nargs="*",
        help="""Training config file with corresponding optional key=value pairs. 
                                       If key presents in the preceding config file, the value in the config
                                       file will be overwritten by the new value """,
    )

    submit_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")
    job_sub_cmd_parser[CMD_SUBMIT_JOB] = submit_parser


def define_list_templates_parser(job_subparser):
    show_jobs_parser = job_subparser.add_parser("list_templates", help="show available job templates")
    show_jobs_parser.add_argument(
        "-d",
        "--job_templates_dir",
        type=str,
        nargs="?",
        default=None,
        help="Job template directory, if not specified, "
        "will search from ~/.nvflare/config.conf and NVFLARE_HOME env. variables",
    )
    show_jobs_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")
    job_sub_cmd_parser[CMD_LIST_TEMPLATES] = show_jobs_parser


def define_variables_parser(job_subparser):
    show_variables_parser = job_subparser.add_parser(
        "show_variables", help="show template variable values in configuration"
    )
    show_variables_parser.add_argument(
        "-j",
        "--job_folder",
        type=str,
        nargs="?",
        default=os.path.join(get_curr_dir(), "current_job"),
        help="job_folder path, default to ./current_job directory",
    )
    show_variables_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")
    job_sub_cmd_parser[CMD_SHOW_VARIABLES] = show_variables_parser


def define_create_job_parser(job_subparser):
    create_parser = job_subparser.add_parser("create", help="create job")
    create_parser.add_argument(
        "-j",
        "--job_folder",
        type=str,
        nargs="?",
        default=os.path.join(get_curr_dir(), "current_job"),
        help="job_folder path, default to ./current_job directory",
    )
    create_parser.add_argument(
        "-w",
        "--template",
        type=str,
        nargs="?",
        default="sag_pt",
        help="""template name or template folder. You can use list_templates to see available jobs from job templates, 
                pick name such as 'sag_pt' as template name. 
                Alternatively, you can use the path to the job template folder, such as job_templates/sag_pt 
                """,
    )
    create_parser.add_argument(
        "-sd",
        "--script_dir",
        type=str,
        nargs="?",
        help="""script directory contains additional related files. 
                                       All files or directories under this directory will be copied over 
                                       to the custom directory.""",
    )
    create_parser.add_argument(
        "-f",
        "--config_file",
        type=str,
        action="append",
        nargs="*",
        help="""Training config file with corresponding optional key=value pairs. 
                                       If key presents in the preceding config file, the value in the config
                                       file will be overwritten by the new value """,
    )
    create_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")
    create_parser.add_argument(
        "-force",
        "--force",
        action="store_true",
        help="force create is on, if -force, " "overwrite existing configuration with newly created configurations",
    )

    job_sub_cmd_parser[CMD_CREATE_JOB] = create_parser


def prepare_job_config(cmd_args, app_names: List[str], tmp_job_dir: Optional[str] = None):
    merged_conf, config_modified = merge_configs_from_cli(cmd_args, app_names)
    need_save_config = config_modified is True or tmp_job_dir is not None

    if tmp_job_dir is None:
        tmp_job_dir = cmd_args.job_folder

    if need_save_config:
        save_merged_configs(merged_conf, cmd_args.job_folder, tmp_job_dir)
    variable_values = filter_indices(merged_conf)
    return variable_values


def has_client_config_file(app_config_dir):
    return any(
        [
            os.path.exists(os.path.join(app_config_dir, f"config_fed_client{postfix}"))
            for postfix in ConfigFormat.extensions()
        ]
    )


def save_merged_configs(app_merged_conf, job_folder, tmp_job_dir):
    for app_name, merged_conf in app_merged_conf.items():
        for file, (config, excluded_key_List, key_indices) in merged_conf.items():
            if job_folder == tmp_job_dir:
                dst_path = file
            else:
                rel_file_path = os.path.relpath(file, job_folder)
                dst_path = os.path.join(tmp_job_dir, rel_file_path)

            root_index = get_root_index(next(iter(key_indices.values()))[0])
            save_config(root_index.value, dst_path)


def prepare_meta_config(cmd_args, target_template_dir, app_names):
    job_folder = cmd_args.job_folder
    job_folder = job_folder[:-1] if job_folder.endswith("/") else job_folder

    job_name = os.path.basename(job_folder)
    meta_files = [f"{JOB_META_BASE_NAME}{postfix}" for postfix in ConfigFormat.extensions()]
    dst_path = None
    for mf in meta_files:
        meta_path = os.path.join(job_folder, mf)
        if os.path.isfile(meta_path):
            dst_path = meta_path
            break

    src_meta_path = os.path.join(target_template_dir, f"{JOB_META_BASE_NAME}.conf")
    if not os.path.isfile(src_meta_path):
        dst_config = load_default_config_template(f"{JOB_META_BASE_NAME}.conf")
    else:
        dst_config = CF.parse_file(src_meta_path)

    # Use existing meta.conf if user already defined it.
    if not dst_path or (dst_path and cmd_args.force):
        dst_config.put("name", job_name)
        dst_path = os.path.join(job_folder, f"{JOB_META_BASE_NAME}.conf")
        save_config(dst_config, dst_path)

    # clean up
    app_names = [DEFAULT_APP_NAME] if not app_names else app_names
    for app_name in app_names:
        config_dir = get_config_dir(job_folder, app_name)
        for mf in meta_files:
            meta_path = os.path.join(config_dir, mf)
            if os.path.isfile(meta_path):
                os.remove(meta_path)


def load_default_config_template(config_file_name: str):
    file_dir = os.path.dirname(__file__)
    # src config here is always pyhocon
    config_template = CF.parse_file(os.path.join(file_dir, f"config/{config_file_name}"))
    return config_template


def dst_app_path(job_folder: str, app_name="app"):
    return os.path.join(job_folder, app_name)


def dst_config_path(job_folder, config_filename, app_name: str = "app"):
    config_dir = get_config_dir(job_folder, app_name)
    dst_path = os.path.join(config_dir, config_filename)
    return dst_path


def get_config_dirs(job_folder: str, app_names: List[str]) -> List[str]:
    config_dirs = []
    if app_names:
        for app_name in app_names:
            config_dirs.append(get_config_dir(job_folder, app_name))
    else:
        config_dirs.append(get_config_dir(job_folder, "app"))

    return config_dirs


def get_config_dir(job_folder: str, app_name: str) -> str:
    app_dir = dst_app_path(job_folder, app_name)
    config_dir = os.path.join(app_dir, "config")
    return config_dir


def convert_args_list_to_dict(kvs: Optional[List[str]] = None) -> dict:
    """
    Convert a list of key-value strings to a dictionary.

    Args:
        kvs (Optional[List[str]]): A list of key-value strings in the format "key=value".

    Returns:
        dict: A dictionary containing the key-value pairs from the input list.
    """
    kv_dict = {}
    if kvs:
        for kv in kvs:
            try:
                key, value = kv.split("=")
                kv_dict[key.strip()] = value.strip()
            except ValueError:
                raise ValueError(f"Invalid key-value pair: '{kv}'")

    return kv_dict


def prepare_job_folder(cmd_args):
    job_folder = cmd_args.job_folder
    if job_folder:
        if not os.path.exists(job_folder):
            os.makedirs(job_folder)
        elif not os.path.isdir(job_folder):
            raise ValueError(f"job_folder '{job_folder}' exits but not directory")
        elif cmd_args.force:
            shutil.rmtree(job_folder)
            os.makedirs(job_folder)

    app_folder = os.path.join(cmd_args.job_folder, "app")
    if os.path.exists(app_folder):
        if cmd_args.force:
            shutil.rmtree(app_folder)
        else:
            print(
                """\nwarning: app directory already exists.
                \nIf you would like to overwrite, use -force option"""
            )


def is_subdir(path, directory):
    # Normalize the paths to avoid issues with different OS formats
    path = os.path.realpath(path)
    directory = os.path.realpath(directory)
    # Check if the directory is a prefix of the path
    return os.path.commonpath([path, directory]) == directory


def prepare_app_scripts(job_folder, app_custom_dirs, cmd_args):
    script_dir = cmd_args.script_dir

    for app_custom_dir in app_custom_dirs:
        if script_dir and len(script_dir.strip()) > 0:
            if os.path.exists(script_dir):
                if script_dir == job_folder or is_subdir(job_folder, script_dir):
                    raise ValueError("job_folder must not be the same or sub directory of script_dir")
                copy_tree(cmd_args.script_dir, app_custom_dir)
                remove_pycache_files(app_custom_dir)
            else:
                raise ValueError(f"{cmd_args.script_dir} doesn't exists")


def prepare_app_dirs(job_folder: str, app_names: List[str]) -> List[str]:
    app_names = ["app"] if not app_names else app_names
    app_custom_dirs = []
    for app_name in app_names:
        app_custom_dir = create_app_dir(job_folder=job_folder, app_name=app_name)
        app_custom_dirs.append(app_custom_dir)

    return app_custom_dirs


def create_app_dir(job_folder, app_name: str = "app"):
    app_dir = os.path.join(job_folder, app_name)
    app_config_dir = os.path.join(app_dir, "config")
    app_custom_dir = os.path.join(app_dir, "custom")
    dirs = [app_dir, app_config_dir, app_custom_dir]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return app_custom_dir

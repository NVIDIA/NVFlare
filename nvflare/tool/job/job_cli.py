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
import pathlib
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
    CONFIG_CONF,
    CONFIG_FILE_BASE_NAME_WO_EXTS,
    JOB_CONFIG_COMP_NAME,
    JOB_CONFIG_FILE_NAME,
    JOB_CONFIG_VAR_NAME,
    JOB_CONFIG_VAR_VALUE,
    JOB_INFO_CLIENT_TYPE,
    JOB_INFO_CLIENT_TYPE_KEY,
    JOB_INFO_CONF,
    JOB_INFO_CONTROLLER_TYPE,
    JOB_INFO_CONTROLLER_TYPE_KEY,
    JOB_INFO_DESC,
    JOB_INFO_DESC_KEY,
    JOB_INFO_KEYS,
    JOB_TEMPLATE,
    JOB_TEMPLATE_CONF,
)
from nvflare.utils.cli_utils import (
    find_job_template_location,
    get_curr_dir,
    get_hidden_nvflare_dir,
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


def build_job_template_indices(job_template_dir: str) -> ConfigTree:
    conf = CF.parse_string("{ templates = {} }")
    config_file_base_names = CONFIG_FILE_BASE_NAME_WO_EXTS
    template_conf = conf.get("templates")
    keys = JOB_INFO_KEYS
    for root, dirs, files in os.walk(job_template_dir):
        config_files = [f for f in files if find_filename_basename(f) in config_file_base_names]
        if len(config_files) > 0:
            info_conf = get_template_info_config(root)
            for key in keys:
                value = info_conf.get(key, "NA") if info_conf else "NA"
                template_name = os.path.basename(root)
                template_conf.put(f"{template_name}.{key}", value)

    return conf


def get_template_registry_file_path():
    filename = JOB_TEMPLATE_CONF
    hidden_nvflare_dir = get_hidden_nvflare_dir()
    file_path = os.path.join(hidden_nvflare_dir, filename)
    return file_path


def get_template_info_config(template_dir):
    info_conf_path = os.path.join(template_dir, JOB_INFO_CONF)
    return CF.parse_file(info_conf_path) if os.path.isfile(info_conf_path) else None


def create_job(cmd_args):
    try:
        prepare_job_folder(cmd_args)
        job_template_dir = find_job_template_location()
        template_index_conf = build_job_template_indices(job_template_dir)
        job_folder = cmd_args.job_folder
        config_dir = get_config_dir(job_folder)

        fmt, real_config_path = ConfigFactory.search_config_format("config_fed_server.conf", [config_dir])
        if real_config_path and not cmd_args.force:
            print(
                f"""\nwarning: configuration files:\n
                    {"config_fed_server.[json|conf|yml]"} already exists.
                \nNot generating the config files. If you would like to overwrite, use -force option"""
            )
            return

        target_template_name = cmd_args.template
        check_template_exists(target_template_name, template_index_conf)
        src = os.path.join(job_template_dir, target_template_name)
        copy_tree(src=src, dst=config_dir)
        prepare_meta_config(cmd_args)
        remove_extra_file(config_dir)
        variable_values = prepare_job_config(cmd_args)
        display_template_variables(job_folder, variable_values)

    except ValueError as e:
        print(f"\nUnable to handle command: {CMD_CREATE_JOB} due to: {e} \n")
        if cmd_args.debug:
            print(traceback.format_exc())
        sub_cmd_parser = job_sub_cmd_parser[CMD_CREATE_JOB]
        if sub_cmd_parser:
            sub_cmd_parser.print_help()


def remove_extra_file(config_dir):
    extra_file = ["info.md", "info.conf"]
    for ef in extra_file:
        file_path = os.path.join(config_dir, ef)
        if os.path.isfile(file_path):
            os.remove(file_path)


def show_variables(cmd_args):
    try:
        if not os.path.isdir(cmd_args.job_folder):
            raise ValueError("required job folder is not specified.")

        config_dir = get_config_dir(cmd_args.job_folder)
        indices = build_config_file_indices(config_dir)
        variable_values = filter_indices(indices_configs=indices)
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
    print(f"{targets=}")
    found = target_template_name in targets

    if not found:
        raise ValueError(
            f"Invalid template name {target_template_name}, "
            f"please check the available templates using nvflare job list_templates"
        )


def display_template_variables(job_folder, variable_values):
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
        job_template_dir = find_job_template_location(cmd_args.job_template_dir)
        job_template_dir = os.path.abspath(job_template_dir)
        template_index_conf = build_job_template_indices(job_template_dir)
        display_available_templates(template_index_conf)

        if job_template_dir:
            update_job_template_dir(job_template_dir)

    except ValueError as e:
        print(f"\nUnable to handle command: {CMD_LIST_TEMPLATES} due to: {e} \n")
        if cmd_args.debug:
            print(traceback.format_exc())
        sub_cmd_parser = job_sub_cmd_parser[CMD_LIST_TEMPLATES]
        if sub_cmd_parser:
            sub_cmd_parser.print_help()


def update_job_template_dir(job_template_dir: str):
    hidden_nvflare_dir = get_hidden_nvflare_dir()
    file_path = os.path.join(hidden_nvflare_dir, CONFIG_CONF)
    config = CF.parse_file(file_path)
    config.put(f"{JOB_TEMPLATE}.path", job_template_dir)
    save_config(config, file_path)


def display_available_templates(template_index_conf):
    print("\nThe following job templates are available: \n")
    template_registry = template_index_conf.get("templates")
    total_length = 120
    left_margin = 1
    print("-" * total_length)
    name_fix_length = 15
    description_fix_length = 60
    controller_type_fix_length = 20
    client_category_fix_length = 20
    name = fix_length_format("name", name_fix_length)
    description = fix_length_format(JOB_INFO_DESC, description_fix_length)
    client_category = fix_length_format(JOB_INFO_CLIENT_TYPE, client_category_fix_length)
    controller_type = fix_length_format(JOB_INFO_CONTROLLER_TYPE, controller_type_fix_length)
    print(" " * left_margin, name, description, controller_type, client_category)
    print("-" * total_length)
    for file_path in sorted(template_registry.keys()):
        name = os.path.basename(file_path)
        template_info = template_registry.get(file_path, None)
        if not template_info:
            template_info = template_registry.get(name)
        name = fix_length_format(name, name_fix_length)
        description = fix_length_format(template_info.get(JOB_INFO_DESC_KEY), description_fix_length)
        client_category = fix_length_format(template_info.get(JOB_INFO_CLIENT_TYPE_KEY), client_category_fix_length)
        controller_type = fix_length_format(template_info.get(JOB_INFO_CONTROLLER_TYPE_KEY), controller_type_fix_length)
        print(" " * left_margin, name, description, controller_type, client_category)
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

        prepare_job_config(cmd_args, temp_job_dir)
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

    admin_user_dir = None
    admin_username = None
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
    submit_parser.add_argument(
        "-a",
        "--app_config",
        type=str,
        nargs="*",
        help="""key=value options will be passed directly to script argument """,
    )

    submit_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")
    job_sub_cmd_parser[CMD_SUBMIT_JOB] = submit_parser


def define_list_templates_parser(job_subparser):
    show_jobs_parser = job_subparser.add_parser("list_templates", help="show available job templates")
    show_jobs_parser.add_argument(
        "-d",
        "--job_template_dir",
        type=str,
        nargs="?",
        default=None,
        help="Job template directory, if not specified, "
        "will search from ./nvflare/config.conf and NVFLARE_HOME env. variables",
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
        help="""template name, use liste_templates to see available jobs from job templates """,
    )
    create_parser.add_argument("-s", "--script", type=str, nargs="?", help="""code script such as train.py""")
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
    create_parser.add_argument(
        "-a",
        "--app_config",
        type=str,
        nargs="*",
        help="""key=value options will be passed directly to script argument """,
    )
    create_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")
    create_parser.add_argument(
        "-force",
        "--force",
        action="store_true",
        help="force create is on, if -force, " "overwrite existing configuration with newly created configurations",
    )

    job_sub_cmd_parser[CMD_CREATE_JOB] = create_parser


def prepare_job_config(cmd_args, tmp_job_dir: Optional[str] = None):
    update_client_app_script(cmd_args)
    merged_conf, config_modified = merge_configs_from_cli(cmd_args)
    need_save_config = config_modified is True or tmp_job_dir is not None

    if tmp_job_dir is None:
        tmp_job_dir = cmd_args.job_folder

    if need_save_config:
        save_merged_configs(merged_conf, tmp_job_dir)

    variable_values = filter_indices(merged_conf)

    return variable_values


def update_client_app_script(cmd_args):
    if cmd_args.app_config:
        print(cmd_args.app_config)
        client_config, config_path = _update_client_app_config_script(cmd_args.job_folder, cmd_args.app_config)
        save_config(client_config, config_path)


def _update_client_app_config_script(job_folder, app_configs: List[str]) -> Tuple[ConfigTree, str]:
    xs = []
    for cli_kv in app_configs:
        tokens = cli_kv.split("=")
        k, v = tokens[0], tokens[1]
        xs.append((k, v))

    config_args = " ".join([f"--{k} {v}" for k, v in xs])
    config_dir = get_config_dir(job_folder)
    config = ConfigFactory.load_config(os.path.join(config_dir, "config_fed_client.xxx"))
    if config.format == ConfigFormat.JSON or config.format == ConfigFormat.OMEGACONF:
        client_config = CF.from_dict(config.to_dict())
    else:
        client_config = config.conf

    client_config.put("app_config", config_args)
    return client_config, config.file_path


def save_merged_configs(merged_conf, tmp_job_dir):
    for file, (config, excluded_key_List, key_indices) in merged_conf.items():
        config_dir = pathlib.Path(tmp_job_dir) / "app" / "config"
        base_filename = os.path.basename(file)
        if base_filename.startswith("meta."):
            config_dir = tmp_job_dir
        dst_path = os.path.join(config_dir, base_filename)
        root_index = get_root_index(next(iter(key_indices.values()))[0])
        save_config(root_index.value, dst_path)


def prepare_model_exchange_config(job_folder: str, force: bool):
    dst_path = dst_config_path(job_folder, "config_exchange.conf")
    if os.path.isfile(dst_path) and not force:
        return

    dst_config = load_src_config_template("config_exchange.conf")
    save_config(dst_config, dst_path)


def prepare_meta_config(cmd_args):
    job_folder = cmd_args.job_folder
    job_folder = job_folder[:-1] if job_folder.endswith("/") else job_folder

    app_name = os.path.basename(job_folder)
    meta_files = ["meta.json", "meta.conf", "meta.yml"]
    dst_path = None
    for mf in meta_files:
        meta_path = os.path.join(job_folder, mf)
        if os.path.isfile(meta_path):
            dst_path = meta_path
            break

    # Use existing meta.conf if user already defined it.
    if not dst_path:
        dst_config = load_src_config_template("meta.conf")
        dst_config.put("name", app_name)
        dst_path = os.path.join(job_folder, "meta.conf")
    else:
        dst_config = CF.from_dict(ConfigFactory.load_config(dst_path).to_dict())

    save_config(dst_config, dst_path)

    # clean up
    config_dir = get_config_dir(job_folder)
    for mf in meta_files:
        meta_path = os.path.join(config_dir, mf)
        if os.path.isfile(meta_path):
            os.remove(meta_path)


def load_src_config_template(config_file_name: str):
    file_dir = os.path.dirname(__file__)
    # src config here is always pyhocon
    config_template = CF.parse_file(os.path.join(file_dir, f"config/{config_file_name}"))
    return config_template


def dst_app_path(job_folder: str):
    return os.path.join(job_folder, "app")


def dst_config_path(job_folder, config_filename):
    config_dir = get_config_dir(job_folder)
    dst_path = os.path.join(config_dir, config_filename)
    return dst_path


def get_config_dir(job_folder):
    app_dir = dst_app_path(job_folder)
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

    app_dir = os.path.join(job_folder, "app")
    app_config_dir = os.path.join(app_dir, "config")
    app_custom_dir = os.path.join(app_dir, "custom")
    dirs = [app_dir, app_config_dir, app_custom_dir]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    if cmd_args.script and len(cmd_args.script.strip()) > 0:
        if os.path.exists(cmd_args.script):
            shutil.copy(cmd_args.script, app_custom_dir)
        else:
            raise ValueError(f"{cmd_args.script} doesn't exists")

    if cmd_args.script_dir and len(cmd_args.script_dir.strip()) > 0:
        if os.path.exists(cmd_args.script_dir):
            copy_tree(cmd_args.script_dir, app_custom_dir)
        else:
            raise ValueError(f"{cmd_args.script_dir} doesn't exists")

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
import datetime
import os
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from functools import partial
from tempfile import mkdtemp
from typing import List, Optional, Tuple

from pyhocon import ConfigFactory as CF
from pyhocon import ConfigTree

from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException
from nvflare.fuel.utils.config import ConfigFormat
from nvflare.fuel.utils.config_factory import ConfigFactory
from nvflare.tool.cli_session import (
    add_startup_kit_selection_args,
    new_cli_session,
    new_cli_session_for_args,
)
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
    load_hidden_config_state,
    save_config,
)

CMD_LIST_TEMPLATES = "list-templates"
CMD_SHOW_VARIABLES = "show-variables"
CMD_CREATE_JOB = "create"
CMD_SUBMIT_JOB = "submit"
CMD_JOB_LIST = "list"
CMD_JOB_META = "meta"
CMD_JOB_ABORT = "abort"
CMD_JOB_CLONE = "clone"
CMD_JOB_DOWNLOAD = "download"
CMD_JOB_DELETE = "delete"

# Job observability commands
CMD_JOB_STATS = "stats"
CMD_JOB_LOGS = "logs"

# Job lifecycle helpers
CMD_JOB_MONITOR = "monitor"
CMD_JOB_LOG_CONFIG = "log-config"
CMD_JOB_LOG_ALIAS = "log"

_JOB_HELP_FORMATTER = partial(argparse.HelpFormatter, max_help_position=24, width=120)
_ACTIVE_STARTUP_KIT_HINT = (
    "Run 'nvflare config list' and 'nvflare config use <id>', pass --kit-id <id> or --startup-kit <path>, "
    "or set NVFLARE_STARTUP_KIT_DIR for automation."
)


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
    for root, _dirs, files in os.walk(template_dir):
        if root != template_dir and (CONFIG_FED_SERVER_CONF in files or CONFIG_FED_CLIENT_CONF in files):
            app_dirs.append(root)

    return app_dirs


def get_app_dirs_from_job_folder(job_folder):
    app_dirs = []
    for root, _dirs, _files in os.walk(job_folder):
        if root != job_folder and (root.endswith("config") or root.endswith("custom")):
            dir_name = os.path.dirname(os.path.relpath(root, job_folder))
            if dir_name:
                app_dirs.append(dir_name)

    return app_dirs


def create_job(cmd_args):
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_CREATE_JOB],
        "nvflare job create",
        [],
        sys.argv[1:],
        deprecated=True,
        deprecated_message="Use 'python job.py --export --export-dir <job_folder>' + 'nvflare job submit -j <job_folder>' instead.",
    )
    from nvflare.tool.cli_output import print_human

    print_human(
        "WARNING: 'nvflare job create' is deprecated. Use 'python job.py --export --export-dir <job_folder>' + "
        "'nvflare job submit -j <job_folder>' instead. Run 'nvflare recipe list' to see available recipes."
    )
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
            from nvflare.tool.cli_output import print_human

            print_human(
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
            shutil.copytree(src=src, dst=app_config_dir, dirs_exist_ok=True)
            remove_extra_files(app_config_dir)
        prepare_meta_config(cmd_args, template_src, app_names)
        app_variable_values = prepare_job_config(cmd_args, app_names)
        display_template_variables(job_folder, app_variable_values)

    except ValueError as e:
        from nvflare.tool.cli_output import output_usage_error, print_human

        if cmd_args.debug:
            print_human(traceback.format_exc())
        output_usage_error(job_sub_cmd_parser[CMD_CREATE_JOB], detail=str(e), exit_code=4)


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
    for root, dirs, _files in os.walk(custom_dir):
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
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_SHOW_VARIABLES],
        "nvflare job show-variables",
        [],
        sys.argv[1:],
        deprecated=True,
        deprecated_message="Use the Job Recipe API instead.",
    )
    from nvflare.tool.cli_output import print_human

    if getattr(cmd_args, "job_sub_cmd", None) == "show_variables":
        print_human("WARNING: 'nvflare job show_variables' is deprecated; use 'nvflare job show-variables' instead.")
    print_human("WARNING: 'nvflare job show-variables' is deprecated. Use the Job Recipe API instead.")
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
        from nvflare.tool.cli_output import output_usage_error, print_human

        if cmd_args.debug:
            print_human(traceback.format_exc())
        output_usage_error(job_sub_cmd_parser[CMD_SHOW_VARIABLES], detail=str(e), exit_code=4)


def check_template_exists(target_template_name, template_index_conf):
    targets = [os.path.basename(key) for key in template_index_conf.get("templates").keys()]
    found = target_template_name in targets

    if not found:
        raise ValueError(
            f"Invalid template name {target_template_name}, "
            f"please check the available templates using nvflare job list-templates"
        )


def display_template_variables(job_folder, app_variable_values):
    from nvflare.tool.cli_output import print_human

    print_human("\nThe following are the variables you can change in the template\n")
    total_length = 135
    left_margin = 1
    print_human("-" * total_length)
    job_folder_header = fix_length_format(f"job folder: {job_folder}", total_length)
    print_human(" " * total_length)
    print_human(" " * left_margin, job_folder_header)
    print_human(" " * total_length)
    print_human("-" * total_length)
    file_name_fix_length = 30
    var_name_fix_length = 30
    var_value_fix_length = 35
    var_comp_fix_length = 35
    file_name = fix_length_format(JOB_CONFIG_FILE_NAME, file_name_fix_length)
    var_name = fix_length_format(JOB_CONFIG_VAR_NAME, var_name_fix_length)
    var_value = fix_length_format(JOB_CONFIG_VAR_VALUE, var_value_fix_length)
    var_comp = fix_length_format(JOB_CONFIG_COMP_NAME, var_comp_fix_length)
    print_human(" " * left_margin, file_name, var_name, var_value, var_comp)
    print_human("-" * total_length)
    for app_name, variable_values in app_variable_values.items():
        if app_name != DEFAULT_APP_NAME and app_name != META_APP_NAME:
            app_header = fix_length_format(f"app: {app_name}", total_length)
            print_human(" " * left_margin, app_header)
            print_human(" " * total_length)

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
                print_human(" " * left_margin, file_name, var_name, var_value, var_comp)

            print_human("")
    print_human("-" * total_length)


def list_templates(cmd_args):
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_LIST_TEMPLATES],
        "nvflare job list-templates",
        [],
        sys.argv[1:],
        deprecated=True,
        deprecated_message="Use 'nvflare recipe list' instead.",
    )
    from nvflare.tool.cli_output import print_human

    if getattr(cmd_args, "job_sub_cmd", None) == "list_templates":
        print_human("WARNING: 'nvflare job list_templates' is deprecated; use 'nvflare job list-templates' instead.")
    print_human("WARNING: 'nvflare job list-templates' is deprecated. Use 'nvflare recipe list' instead.")
    try:
        job_templates_dir = find_job_templates_location(cmd_args.job_templates_dir)
        job_templates_dir = os.path.abspath(job_templates_dir)
        template_index_conf = build_job_template_indices(job_templates_dir)
        display_available_templates(template_index_conf)

        if job_templates_dir:
            update_job_templates_dir(job_templates_dir)

    except ValueError as e:
        from nvflare.tool.cli_output import output_usage_error, print_human

        if cmd_args.debug:
            print_human(traceback.format_exc())
        output_usage_error(job_sub_cmd_parser[CMD_LIST_TEMPLATES], detail=str(e), exit_code=4)


def update_job_templates_dir(job_templates_dir: str):
    config_file_path, nvflare_config, _migration_needed = load_hidden_config_state()
    if nvflare_config is None:
        from pyhocon import ConfigFactory as CF

        nvflare_config = CF.parse_string("{}")

    config = create_job_template_config(nvflare_config, job_templates_dir)
    save_config(config, config_file_path)


def display_available_templates(template_index_conf):
    from nvflare.tool.cli_output import print_human

    print_human("\nThe following job templates are available: \n")
    template_registry = template_index_conf.get("templates")
    total_length = 120
    left_margin = 1
    print_human("-" * total_length)
    name_fix_length = 20
    description_fix_length = 60
    controller_type_fix_length = 17
    execution_api_type_fix_length = 23
    name = fix_length_format("name", name_fix_length)
    description = fix_length_format(JOB_INFO_DESC, description_fix_length)
    execution_api_type = fix_length_format(JOB_INFO_EXECUTION_API_TYPE, execution_api_type_fix_length)
    controller_type = fix_length_format(JOB_INFO_CONTROLLER_TYPE, controller_type_fix_length)
    print_human(" " * left_margin, name, description, controller_type, execution_api_type)
    print_human("-" * total_length)
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
        print_human(" " * left_margin, name, description, controller_type, execution_api_type)
    print_human("-" * total_length)


def fix_length_format(name: str, name_fix_length: int):
    return f"{name[:name_fix_length]:{name_fix_length}}"


def submit_job(cmd_args):
    from nvflare.tool.cli_schema import handle_schema_flag

    if job_sub_cmd_parser[CMD_SUBMIT_JOB] is None:
        root_parser = argparse.ArgumentParser(prog="nvflare job")
        root_subparser = root_parser.add_subparsers(dest="job_sub_cmd")
        define_submit_job_parser(root_subparser)

    handle_schema_flag(
        job_sub_cmd_parser[CMD_SUBMIT_JOB],
        "nvflare job submit",
        ["nvflare config use admin@nvidia.com", "nvflare job submit -j ./my_job"],
        sys.argv[1:],
    )

    def _has_job_meta(path: str) -> bool:
        for ext in (".json", ".conf", ".yml", ".yaml"):
            if os.path.isfile(os.path.join(path, f"meta{ext}")):
                return True
        return False

    def _has_server_config(path: str) -> bool:
        config_dir = os.path.join(path, "app", "config")
        for ext in (".json", ".conf", ".yml", ".yaml"):
            if os.path.isfile(os.path.join(config_dir, f"config_fed_server{ext}")):
                return True
        return False

    def _resolve_job_folder(path: str) -> str:
        if _has_job_meta(path) and _has_server_config(path):
            return path

        subdirs = []
        for name in os.listdir(path):
            if name.startswith("."):
                continue
            full = os.path.join(path, name)
            if os.path.isdir(full):
                subdirs.append(full)

        if len(subdirs) == 1:
            candidate = subdirs[0]
            if _has_job_meta(candidate) and _has_server_config(candidate):
                from nvflare.tool.cli_output import is_json_mode, print_human

                if not is_json_mode():
                    print_human(f"Using job folder: {candidate}")
                return candidate

        return path

    temp_job_dir = None
    try:
        if not os.path.isdir(cmd_args.job_folder):
            raise ValueError(f"invalid job folder: {cmd_args.job_folder}")

        job_folder = _resolve_job_folder(cmd_args.job_folder)

        temp_job_dir = mkdtemp()
        shutil.copytree(job_folder, temp_job_dir, dirs_exist_ok=True)

        app_dirs = get_app_dirs_from_job_folder(job_folder)
        app_names = [os.path.basename(f) for f in app_dirs]
        app_names = app_names if app_names else [DEFAULT_APP_NAME]

        prepare_job_config(cmd_args, app_names, temp_job_dir)
        internal_submit_job(None, None, temp_job_dir, cmd_args)

    except ValueError as e:
        from nvflare.tool.cli_output import output_usage_error, print_human

        if cmd_args.debug:
            print_human(traceback.format_exc())
        output_usage_error(job_sub_cmd_parser[CMD_SUBMIT_JOB], detail=str(e), exit_code=4)
    finally:
        if temp_job_dir:
            if cmd_args.debug:
                from nvflare.tool.cli_output import print_human

                print_human(f"in debug mode, job configurations can be examined in temp job directory '{temp_job_dir}'")
            else:
                shutil.rmtree(temp_job_dir)


def _resolve_admin_user_and_dir_from_startup_kit(startup_kit_dir: str) -> Tuple[str, str]:
    from nvflare.tool.kit.kit_config import resolve_admin_user_and_dir_from_startup_kit

    return resolve_admin_user_and_dir_from_startup_kit(startup_kit_dir)


def internal_submit_job(admin_user_dir, username, temp_job_dir, cmd_args=None):
    from nvflare.fuel.flare_api.api_spec import AuthorizationError, InternalError, InvalidJobDefinition, NoConnection
    from nvflare.tool.cli_output import is_json_mode, output_error, output_ok, print_human

    if not is_json_mode():
        print_human("trying to connect to the server")
    study = getattr(cmd_args, "study", "default") if cmd_args else "default"

    sess = _get_session(args=cmd_args, admin_user_dir=admin_user_dir, username=username, study=study)
    try:
        try:
            job_id = sess.submit_job(temp_job_dir)
        except InvalidJobDefinition as e:
            output_error("JOB_INVALID", exit_code=1, detail=str(e))
            raise SystemExit(1)
        except AuthorizationError as e:
            output_error("AUTH_FAILED", exit_code=2, detail=str(e))
            raise SystemExit(2)
        except InternalError as e:
            output_error("INTERNAL_ERROR", exit_code=5, detail=str(e))
            raise SystemExit(5)
        except NoConnection as e:
            output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
            raise SystemExit(2)
        output_ok({"job_id": job_id})
    finally:
        sess.close()


job_sub_cmd_handlers = {
    CMD_CREATE_JOB: create_job,
    CMD_SUBMIT_JOB: submit_job,
    CMD_LIST_TEMPLATES: list_templates,
    CMD_SHOW_VARIABLES: show_variables,
    CMD_JOB_LIST: None,
    CMD_JOB_META: None,
    CMD_JOB_ABORT: None,
    CMD_JOB_CLONE: None,
    CMD_JOB_DOWNLOAD: None,
    CMD_JOB_DELETE: None,
    CMD_JOB_STATS: None,
    CMD_JOB_LOGS: None,
    CMD_JOB_MONITOR: None,
    CMD_JOB_LOG_CONFIG: None,
}

job_sub_cmd_parser = {
    CMD_CREATE_JOB: None,
    CMD_SUBMIT_JOB: None,
    CMD_LIST_TEMPLATES: None,
    CMD_SHOW_VARIABLES: None,
    CMD_JOB_LIST: None,
    CMD_JOB_META: None,
    CMD_JOB_ABORT: None,
    CMD_JOB_CLONE: None,
    CMD_JOB_DOWNLOAD: None,
    CMD_JOB_DELETE: None,
    CMD_JOB_STATS: None,
    CMD_JOB_LOGS: None,
    CMD_JOB_MONITOR: None,
    CMD_JOB_LOG_CONFIG: None,
}


def handle_job_cli_cmd(cmd_args):
    sub_cmd = {
        "list_templates": CMD_LIST_TEMPLATES,
        "show_variables": CMD_SHOW_VARIABLES,
    }.get(cmd_args.job_sub_cmd, cmd_args.job_sub_cmd)
    cmd_args.job_sub_cmd = sub_cmd
    job_cmd_handler = job_sub_cmd_handlers.get(sub_cmd, None)
    if job_cmd_handler:
        job_cmd_handler(cmd_args)
    elif cmd_args.job_sub_cmd is None:
        raise CLIUnknownCmdException("\n no job subcommand provided. \n")
    else:
        raise CLIUnknownCmdException("\n invalid command. \n")


def def_job_cli_parser(sub_cmd):
    cmd = "job"
    parser = sub_cmd.add_parser(cmd, help="submit, manage, and monitor FL jobs", formatter_class=_JOB_HELP_FORMATTER)
    job_subparser = parser.add_subparsers(title="job subcommands", metavar="", dest="job_sub_cmd")
    define_submit_job_parser(job_subparser)
    define_job_monitor_parser(job_subparser)
    define_list_jobs_parser(job_subparser)
    define_abort_job_parser(job_subparser)
    define_job_meta_parser(job_subparser)
    define_job_logs_parser(job_subparser)
    define_job_log_parser(job_subparser)
    define_job_stats_parser(job_subparser)
    define_download_job_parser(job_subparser)
    define_clone_job_parser(job_subparser)
    define_delete_job_parser(job_subparser)
    define_list_templates_parser(job_subparser)
    define_create_job_parser(job_subparser)
    define_variables_parser(job_subparser)

    return {cmd: parser}


def define_submit_job_parser(job_subparser):
    submit_parser = job_subparser.add_parser("submit", help="submit job")
    submit_parser.add_argument(
        "-j",
        "--job-folder",
        "--job_folder",  # backward compat
        dest="job_folder",
        type=str,
        nargs="?",
        default=os.path.join(get_curr_dir(), "current_job"),
        help="job folder path, default to ./current_job directory",
    )
    submit_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")
    submit_parser.add_argument("--study", type=str, default="default", help="study to submit the job to")
    add_startup_kit_selection_args(submit_parser)
    submit_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_SUBMIT_JOB] = submit_parser


def define_list_templates_parser(job_subparser):
    show_jobs_parser = job_subparser.add_parser(
        CMD_LIST_TEMPLATES,
        aliases=["list_templates"],
        help="[DEPRECATED] use 'nvflare recipe list'",
    )
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
    show_jobs_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_LIST_TEMPLATES] = show_jobs_parser


def define_variables_parser(job_subparser):
    show_variables_parser = job_subparser.add_parser(
        CMD_SHOW_VARIABLES,
        aliases=["show_variables"],
        help="[DEPRECATED] use 'nvflare recipe list' or the Job Recipe API",
    )
    show_variables_parser.add_argument(
        "-j",
        "--job-folder",
        "--job_folder",
        type=str,
        nargs="?",
        default=os.path.join(get_curr_dir(), "current_job"),
        help="job folder path, default to ./current_job directory",
    )
    show_variables_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")
    show_variables_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_SHOW_VARIABLES] = show_variables_parser


def define_create_job_parser(job_subparser):
    create_parser = job_subparser.add_parser(
        "create",
        help="[DEPRECATED] use 'python job.py --export --export-dir <job_folder>' + 'nvflare job submit -j <job_folder>'",
    )
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
        help="""template name or template folder. You can use list-templates to see available jobs from job templates,
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
    create_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")

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
                key, value = kv.split("=", 1)
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
            from nvflare.tool.cli_output import print_human

            print_human(
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
                shutil.copytree(cmd_args.script_dir, app_custom_dir, dirs_exist_ok=True)
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


# ---------------------------------------------------------------------------
# Section 3: New Job Lifecycle Commands
# ---------------------------------------------------------------------------


def _get_session(args=None, admin_user_dir=None, username=None, study="default"):
    """Create a secure session using command selectors, env, or active startup kit."""
    from nvflare.tool.cli_output import get_connect_timeout, output_error

    timeout = get_connect_timeout()

    if admin_user_dir is None and username is None:
        try:
            return new_cli_session_for_args(
                args=args,
                timeout=timeout,
                study=study,
                debug=_get_arg_value(args, "debug", False),
            )
        except ValueError as e:
            output_error(
                "STARTUP_KIT_MISSING",
                exit_code=2,
                detail=str(e),
                hint=getattr(e, "hint", None) or _ACTIVE_STARTUP_KIT_HINT,
            )
            raise SystemExit(2)

    if admin_user_dir is None or username is None:
        try:
            from nvflare.tool.cli_session import resolve_admin_user_and_dir_for_args

            u, d = resolve_admin_user_and_dir_for_args(args)
        except ValueError as e:
            output_error(
                "STARTUP_KIT_MISSING",
                exit_code=2,
                detail=str(e),
                hint=getattr(e, "hint", None) or _ACTIVE_STARTUP_KIT_HINT,
            )
            raise SystemExit(2)
        if username is None:
            username = u
        if admin_user_dir is None:
            admin_user_dir = d

    return new_cli_session(
        username=username,
        startup_kit_location=admin_user_dir,
        timeout=timeout,
        study=study,
    )


@contextmanager
def _session(args=None, admin_user_dir=None, username=None, study="default"):
    sess = _get_session(args=args, admin_user_dir=admin_user_dir, username=username, study=study)
    try:
        yield sess
    finally:
        if sess is not None:
            sess.close()


def _get_arg_value(args, name, default=None):
    if args is None:
        return default
    try:
        return vars(args).get(name, default)
    except TypeError:
        return getattr(args, name, default)


def _has_scoped_startup_kit_args(args) -> bool:
    return bool(_get_arg_value(args, "kit_id") or _get_arg_value(args, "startup_kit"))


def _job_session_for_args(cmd_args=None, study="default"):
    if study != "default" or _has_scoped_startup_kit_args(cmd_args):
        return _session(args=cmd_args, study=study)
    return _session()


def cmd_job_list(cmd_args):
    from nvflare.fuel.flare_api.api_spec import AuthenticationError, NoConnection
    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_LIST],
        "nvflare job list",
        [
            "nvflare job list",
            "nvflare job list -n cifar -m 10",
            "nvflare job list --study all",
        ],
        sys.argv[1:],
    )

    study = getattr(cmd_args, "study", "default")

    try:
        with _job_session_for_args(cmd_args, study=study) as sess:
            jobs = sess.list_jobs(
                name_prefix=getattr(cmd_args, "name", None),
                id_prefix=getattr(cmd_args, "id", None),
                reverse=getattr(cmd_args, "reverse", False),
                limit=getattr(cmd_args, "max", None),
            )
    except AuthenticationError:
        raise
    except NoConnection as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    for j in jobs:
        if "study" not in j:
            j["study"] = study

    output_ok(jobs)


def cmd_job_meta(cmd_args):
    from nvflare.fuel.flare_api.api_spec import AuthenticationError, JobNotFound, NoConnection
    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_META],
        "nvflare job meta",
        ["nvflare job meta <job_id>"],
        sys.argv[1:],
    )

    try:
        with _job_session_for_args(cmd_args) as sess:
            meta = sess.get_job_meta(cmd_args.job_id)
    except JobNotFound:
        output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        return
    except AuthenticationError:
        raise
    except NoConnection as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    if meta is None:
        output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        return
    else:
        output_ok(meta)


def cmd_job_abort(cmd_args):
    from nvflare.fuel.flare_api.api_spec import AuthenticationError, JobNotFound, JobNotRunning, NoConnection
    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_ABORT],
        "nvflare job abort",
        ["nvflare job abort <job_id>", "nvflare job abort <job_id> --force"],
        sys.argv[1:],
    )

    if not cmd_args.force:
        if not sys.stdin.isatty():
            output_error("INVALID_ARGS", exit_code=4, detail="use --force in non-interactive mode")
            raise SystemExit(4)
        from nvflare.tool.cli_output import print_human, prompt_yn

        if not prompt_yn(f"Abort job '{cmd_args.job_id}'?"):
            print_human("Aborted.")
            return

    try:
        with _job_session_for_args(cmd_args) as sess:
            sess.abort_job(cmd_args.job_id)
    except JobNotFound:
        output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        return
    except JobNotRunning:
        output_error("JOB_NOT_RUNNING", job_id=cmd_args.job_id)
        return
    except AuthenticationError:
        raise
    except NoConnection as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    output_ok({"job_id": cmd_args.job_id, "status": "ABORTED"})


def cmd_job_clone(cmd_args):
    from nvflare.fuel.flare_api.api_spec import AuthenticationError, JobNotFound, NoConnection
    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_CLONE],
        "nvflare job clone",
        ["nvflare job clone <job_id>"],
        sys.argv[1:],
    )

    try:
        with _job_session_for_args(cmd_args) as sess:
            new_job_id = sess.clone_job(cmd_args.job_id)
    except JobNotFound:
        output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        return
    except AuthenticationError:
        raise
    except NoConnection as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    output_ok({"source_job_id": cmd_args.job_id, "new_job_id": new_job_id})


def cmd_job_download(cmd_args):
    from nvflare.fuel.flare_api.api_spec import AuthenticationError, JobNotFound, NoConnection
    from nvflare.tool.cli_output import output_error, output_ok, print_human
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_DOWNLOAD],
        "nvflare job download",
        ["nvflare job download <job_id>", "nvflare job download <job_id> -o /path/to/results"],
        sys.argv[1:],
    )

    destination = os.path.abspath(getattr(cmd_args, "output_dir", "./"))
    print_human(f"Downloading job {cmd_args.job_id} ...")
    try:
        with _job_session_for_args(cmd_args) as sess:
            path = sess.download_job_result(cmd_args.job_id, destination)
    except JobNotFound:
        output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        return
    except AuthenticationError:
        raise
    except NoConnection as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    final_path = path or destination
    print_human(f"Job result downloaded to: {final_path}")
    output_ok({"job_id": cmd_args.job_id, "path": final_path})


def cmd_job_delete(cmd_args):
    from nvflare.fuel.flare_api.api_spec import AuthenticationError, JobNotFound, NoConnection
    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_DELETE],
        "nvflare job delete",
        ["nvflare job delete <job_id>", "nvflare job delete <job_id> --force"],
        sys.argv[1:],
    )

    if not cmd_args.force:
        if not sys.stdin.isatty():
            output_error("INVALID_ARGS", exit_code=4, detail="use --force in non-interactive mode")
            raise SystemExit(4)
        from nvflare.tool.cli_output import print_human, prompt_yn

        if not prompt_yn(f"Delete job '{cmd_args.job_id}'?"):
            print_human("Cancelled.")
            return

    try:
        with _job_session_for_args(cmd_args) as sess:
            sess.delete_job(cmd_args.job_id)
    except JobNotFound:
        output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        return
    except AuthenticationError:
        raise
    except NoConnection as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    output_ok({"job_id": cmd_args.job_id})


# ---------------------------------------------------------------------------
# Parser definitions for new commands
# ---------------------------------------------------------------------------


def define_list_jobs_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_LIST, help="list jobs on the server")
    p.add_argument("-n", "--name", type=str, default=None, help="filter by name prefix")
    p.add_argument("-i", "--id", type=str, default=None, help="filter by job ID prefix")
    p.add_argument("-r", "--reverse", action="store_true", default=False, help="reverse sort order")
    p.add_argument("-m", "--max", type=int, default=None, help="max results to return")
    p.add_argument("--study", type=str, default="default", help="study to list jobs from")
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_LIST] = p
    job_sub_cmd_handlers[CMD_JOB_LIST] = cmd_job_list


def define_job_meta_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_META, help="get metadata for a job")
    p.add_argument("job_id", type=str, help="job ID")
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_META] = p
    job_sub_cmd_handlers[CMD_JOB_META] = cmd_job_meta


def define_abort_job_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_ABORT, help="abort a running job")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument("--force", action="store_true", help="skip confirmation prompt")
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_ABORT] = p
    job_sub_cmd_handlers[CMD_JOB_ABORT] = cmd_job_abort


def define_clone_job_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_CLONE, help="clone an existing job")
    p.add_argument("job_id", type=str, help="job ID to clone")
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_CLONE] = p
    job_sub_cmd_handlers[CMD_JOB_CLONE] = cmd_job_clone


def define_download_job_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_DOWNLOAD, help="download job result")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="./",
        help="destination directory",
    )
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_DOWNLOAD] = p
    job_sub_cmd_handlers[CMD_JOB_DOWNLOAD] = cmd_job_download


def define_delete_job_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_DELETE, help="delete a job")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument("--force", action="store_true", help="skip confirmation prompt")
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_DELETE] = p
    job_sub_cmd_handlers[CMD_JOB_DELETE] = cmd_job_delete


_TERMINAL_JOB_STATES = {
    "FINISHED_OK",
    "FINISHED_EXCEPTION",
    "ABORTED",
    "ABANDONED",
    "FAILED",
}


def cmd_job_stats(cmd_args):
    from nvflare.fuel.flare_api.api_spec import (
        AuthenticationError,
        JobNotFound,
        NoConnection,
    )
    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_STATS],
        "nvflare job stats",
        ["nvflare job stats abc123", "nvflare job stats abc123 --site server"],
        sys.argv[1:],
    )

    site = getattr(cmd_args, "site", "all")
    if site == "all":
        target_type = "all"
        targets = None
    elif site == "server":
        target_type = "server"
        targets = None
    else:
        target_type = "client"
        targets = [site]

    try:
        with _job_session_for_args(cmd_args) as sess:
            result = sess.show_stats(cmd_args.job_id, target_type, targets)
    except JobNotFound:
        output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        return
    except AuthenticationError:
        raise
    except NoConnection as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    output_ok({"job_id": cmd_args.job_id, "stats": result})


def cmd_job_logs(cmd_args):
    from nvflare.fuel.flare_api.api_spec import AuthenticationError, JobNotFound, NoConnection
    from nvflare.tool.cli_output import is_json_mode, output_error, output_ok, print_human
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_LOGS],
        "nvflare job logs",
        [
            "nvflare job logs abc123",
            "nvflare job logs abc123 --site site-1",
            "nvflare job logs abc123 --site all",
        ],
        sys.argv[1:],
    )

    site = getattr(cmd_args, "site", "server")

    try:
        with _job_session_for_args(cmd_args) as sess:
            result = sess.get_job_logs(cmd_args.job_id, target=site)
    except JobNotFound:
        output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        return
    except AuthenticationError:
        raise
    except NoConnection as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return
    except Exception as e:
        output_error("INTERNAL_ERROR", exit_code=5, detail=str(e))
        return

    logs = result.get("logs", {})
    unavailable = result.get("unavailable", {})
    if site != "all" and site != "server" and site in unavailable and site not in logs:
        output_error(
            "LOG_NOT_FOUND",
            exit_code=1,
            site=site,
            detail=unavailable.get(site),
        )
        return

    payload = {"job_id": cmd_args.job_id, "target": site, "logs": logs}
    if unavailable:
        payload["unavailable"] = unavailable
    if not is_json_mode():
        _print_job_logs_human(site, logs, unavailable, print_human)
        return
    output_ok(payload)


def _print_job_logs_human(target: str, logs: dict, unavailable: dict, print_func):
    def _print_log_text(text):
        if text:
            print_func(text, end="" if text.endswith("\n") else "\n")
        else:
            print_func("(no log content)")

    if target != "all" and target in logs and len(logs) == 1 and not unavailable:
        _print_log_text(logs[target])
        return

    site_names = list(logs.keys())
    if "server" in site_names:
        site_names.remove("server")
        site_names.insert(0, "server")

    for index, site_name in enumerate(site_names):
        if index:
            print_func()
        print_func(f"===== {site_name} =====")
        _print_log_text(logs[site_name])

    if unavailable:
        if logs:
            print_func()
        print_func("Unavailable logs:", file=sys.stderr)
        for site_name in sorted(unavailable):
            print_func(f"{site_name}: {unavailable[site_name]}", file=sys.stderr)


def define_job_stats_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_STATS, help="show running job statistics")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument("--site", default="all", help="target site name or all")
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_STATS] = p
    job_sub_cmd_handlers[CMD_JOB_STATS] = cmd_job_stats


def define_job_logs_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_LOGS, help="retrieve job logs from the server-side log store")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument(
        "--sites",
        "--site",
        dest="site",
        default="server",
        help="target site name, server, or all. Client logs must have been streamed to the server.",
    )
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_LOGS] = p
    job_sub_cmd_handlers[CMD_JOB_LOGS] = cmd_job_logs


def _summarize_monitor_meta(meta: dict, job_meta_key_cls) -> dict:
    if not meta:
        return {}
    fields = {
        "job_name": job_meta_key_cls.JOB_NAME.value,
        "status": job_meta_key_cls.STATUS.value,
        "submit_time": job_meta_key_cls.SUBMIT_TIME_ISO.value,
        "start_time": job_meta_key_cls.START_TIME.value,
        "duration": job_meta_key_cls.DURATION.value,
        "study": job_meta_key_cls.STUDY.value,
        "submitter_name": job_meta_key_cls.SUBMITTER_NAME.value,
        "submitter_org": job_meta_key_cls.SUBMITTER_ORG.value,
        "submitter_role": job_meta_key_cls.SUBMITTER_ROLE.value,
    }
    summary = {}
    for out_key, meta_key in fields.items():
        value = meta.get(meta_key)
        if value not in (None, ""):
            summary[out_key] = value
    return summary


def _parse_monitor_start_ts(meta: dict, start_time_key: str, submit_time_iso_key: str) -> float:
    if not meta:
        return None
    start_time = meta.get(start_time_key)
    if start_time:
        try:
            return datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f").timestamp()
        except Exception:
            pass
    submit_time_iso = meta.get(submit_time_iso_key)
    if submit_time_iso:
        try:
            return datetime.datetime.fromisoformat(submit_time_iso).timestamp()
        except Exception:
            pass
    return None


def _parse_monitor_duration_seconds(value) -> float:
    if value is None or value == "" or value == "N/A":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    parts = value.split(":")
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        if len(parts) == 1:
            return float(parts[0])
    except Exception:
        return None
    return None


def _build_monitor_key_aliases(extra_metrics: list) -> dict:
    aliases = {
        "round": ["round", "global_round", "current_round", "iteration", "iter", "epoch"],
        "accuracy": ["accuracy", "acc", "val_acc", "test_acc"],
        "loss": ["loss", "train_loss", "val_loss", "test_loss"],
    }
    for metric in extra_metrics:
        aliases[metric] = [metric]
    return aliases


def _extract_monitor_metrics(stats: dict, key_aliases: dict) -> dict:
    if not isinstance(stats, dict):
        return {}

    def _find_key(d: dict, keys: list):
        for k in keys:
            if k in d and isinstance(d[k], (int, float, str)):
                return d[k]
        return None

    def _search(d: dict, keys: list):
        if not isinstance(d, dict):
            return None
        value = _find_key(d, keys)
        if value is not None:
            return value
        for v in d.values():
            if isinstance(v, dict):
                found = _search(v, keys)
                if found is not None:
                    return found
        return None

    metrics = {}
    for out_key, aliases in key_aliases.items():
        value = _search(stats, aliases)
        if value is not None:
            metrics[out_key] = value
    return metrics


def _make_monitor_state() -> dict:
    return {
        "last_status": None,
        "last_meta": None,
        "last_emit_ts": 0.0,
        "last_stats": None,
        "last_stats_raw": None,
        "last_stats_ts": 0.0,
    }


def _refresh_monitor_stats(sess, job_id: str, state: dict, stats_target: str, key_aliases: dict):
    try:
        stats = sess.show_stats(job_id, stats_target, None)
        state["last_stats_raw"] = stats
        state["last_stats"] = _extract_monitor_metrics(stats, key_aliases)
    except Exception:
        state["last_stats"] = None
        state["last_stats_raw"] = None


def _emit_monitor_progress(job_id: str, job_meta: dict, state: dict, now: float, start: float, start_ts):
    from nvflare.apis.job_def import JobMetaKey
    from nvflare.tool.cli_output import print_human

    status = job_meta.get("status", "UNKNOWN") if job_meta else "UNKNOWN"
    summary = _summarize_monitor_meta(job_meta, JobMetaKey)
    name = summary.get("job_name")
    elapsed_base = start_ts if start_ts is not None else start
    elapsed = round(now - elapsed_base, 1)
    message_parts = []
    if state["last_status"] is None:
        message_parts.append(f"job_id: {job_id}")
        if name:
            message_parts.append(f"name: {name}")
        submit_time = summary.get("submit_time")
        if submit_time:
            message_parts.append(f"submit_time: {submit_time}")
    message_parts.append(f"status: {status}")
    message_parts.append(f"elapsed_s: {elapsed}")
    metrics = state.get("last_stats") or {}
    if metrics:
        metric_str = " ".join(f"{k}={v}" for k, v in metrics.items())
        message_parts.append(f"metrics: {metric_str}")
    print_human(" ".join(message_parts))
    state["last_status"] = status
    state["last_emit_ts"] = now


def _build_monitor_status_callback(
    start: float, start_ts_holder: dict, emit_interval: int, stats_interval: int, stats_target: str, key_aliases: dict
):
    def _status_cb(sess, job_id, job_meta, state):
        from nvflare.apis.job_def import JobMetaKey

        state["last_meta"] = job_meta
        status = job_meta.get("status", "UNKNOWN") if job_meta else "UNKNOWN"
        now = time.time()
        if start_ts_holder["value"] is None:
            start_ts_holder["value"] = _parse_monitor_start_ts(
                job_meta, JobMetaKey.START_TIME.value, JobMetaKey.SUBMIT_TIME_ISO.value
            )
        if status in ("RUNNING", "DISPATCHED") and now - state["last_stats_ts"] >= stats_interval:
            _refresh_monitor_stats(sess, job_id, state, stats_target, key_aliases)
            state["last_stats_ts"] = now
        if status != state["last_status"] or now - state["last_emit_ts"] >= emit_interval:
            _emit_monitor_progress(job_id, job_meta, state, now, start, start_ts_holder["value"])
        return status not in _TERMINAL_JOB_STATES

    return _status_cb


def _build_monitor_output_data(
    job_id: str, meta: dict, start: float, start_ts, cb_state: dict, json_mode: bool
) -> dict:
    from nvflare.apis.job_def import JobMetaKey

    meta_duration_s = _parse_monitor_duration_seconds(meta.get("duration") if meta else None)
    if meta_duration_s is not None:
        duration = round(meta_duration_s, 1)
    elif start_ts is not None:
        duration = round(time.time() - start_ts, 1)
    else:
        duration = round(time.time() - start, 1)

    data = {
        "job_id": job_id,
        "status": meta.get("status", "UNKNOWN"),
        "duration_s": duration,
        "job_meta": _summarize_monitor_meta(meta, JobMetaKey),
        "last_stats": cb_state.get("last_stats"),
    }
    if json_mode:
        data["stats_raw"] = cb_state.get("last_stats_raw")
    return data


def cmd_job_monitor(cmd_args):
    from nvflare.fuel.flare_api.api_spec import AuthenticationError, JobNotFound, MonitorReturnCode, NoConnection
    from nvflare.tool.cli_output import is_json_mode, output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_MONITOR],
        "nvflare job monitor",
        [
            "nvflare job monitor abc123",
            "nvflare job monitor abc123 --timeout 3600",
            "nvflare job monitor abc123 --study cancer",
        ],
        sys.argv[1:],
    )

    study = _get_arg_value(cmd_args, "study", "default")
    start = time.time()
    start_ts_holder = {"value": None}
    timeout = getattr(cmd_args, "timeout", 0)
    interval = getattr(cmd_args, "interval", 2)
    cb_state = _make_monitor_state()
    emit_interval = max(interval, 5)
    stats_interval = max(interval, 10)
    stats_target = getattr(cmd_args, "stats_target", "server")
    extra_metrics = [m for m in (getattr(cmd_args, "metrics", None) or []) if m]

    key_aliases = _build_monitor_key_aliases(extra_metrics)
    status_cb = _build_monitor_status_callback(
        start=start,
        start_ts_holder=start_ts_holder,
        emit_interval=emit_interval,
        stats_interval=stats_interval,
        stats_target=stats_target,
        key_aliases=key_aliases,
    )

    try:
        with _job_session_for_args(cmd_args, study=study) as sess:
            rc, meta = sess.monitor_job_and_return_job_meta(
                cmd_args.job_id,
                timeout=timeout,
                poll_interval=interval,
                cb=status_cb,
                state=cb_state,
            )
    except JobNotFound:
        output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        return
    except (AuthenticationError, NoConnection):
        raise
    except Exception as e:
        output_error("INTERNAL_ERROR", exit_code=5, detail=str(e))
        return

    if rc == MonitorReturnCode.TIMEOUT:
        output_error("TIMEOUT", exit_code=3, detail="job did not reach terminal state within timeout")
        return

    if rc == MonitorReturnCode.ENDED_BY_CB:
        meta = cb_state.get("last_meta")
        if meta is None:
            output_error("INTERNAL_ERROR", exit_code=5, detail="monitoring stopped before job metadata was available")
            return

    if not meta:
        output_error("INTERNAL_ERROR", exit_code=5, detail="monitoring returned no job metadata")
        return

    status = meta.get("status", "UNKNOWN")
    data = _build_monitor_output_data(
        job_id=cmd_args.job_id,
        meta=meta,
        start=start,
        start_ts=start_ts_holder["value"],
        cb_state=cb_state,
        json_mode=is_json_mode(),
    )

    if status in ("FAILED", "FINISHED_EXCEPTION", "ABORTED", "ABANDONED"):
        error_envelopes = {
            "FAILED": (
                "JOB_FAILED",
                "Use 'nvflare job logs <job_id>' and 'nvflare job meta <job_id>' to inspect the failure.",
            ),
            "FINISHED_EXCEPTION": (
                "JOB_FINISHED_EXCEPTION",
                "Use 'nvflare job logs <job_id>' and 'nvflare job meta <job_id>' to inspect the failure.",
            ),
            "ABORTED": (
                "JOB_ABORTED",
                "Use 'nvflare job meta <job_id>' to see abort details.",
            ),
            "ABANDONED": (
                "JOB_ABANDONED",
                "Use 'nvflare job meta <job_id>' to inspect the abandonment details.",
            ),
        }
        error_code, hint = error_envelopes[status]
        output_error(error_code, exit_code=1, hint=hint, data=data, job_id=cmd_args.job_id)
    else:
        output_ok(data)


def cmd_job_log(cmd_args):
    from nvflare.fuel.flare_api.api_spec import (
        AuthenticationError,
        AuthorizationError,
        InternalError,
        InvalidTarget,
        JobNotFound,
        NoConnection,
        NoReply,
    )
    from nvflare.tool.cli_output import output_error, output_ok, output_usage_error
    from nvflare.tool.cli_schema import handle_schema_flag

    invoked_sub_cmd = CMD_JOB_LOG_CONFIG
    if len(sys.argv) > 2 and sys.argv[1] == "job" and sys.argv[2] in (CMD_JOB_LOG_CONFIG, CMD_JOB_LOG_ALIAS):
        invoked_sub_cmd = sys.argv[2]

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_LOG_CONFIG],
        f"nvflare job {invoked_sub_cmd}",
        [
            f"nvflare job {invoked_sub_cmd} abc123 DEBUG",
            f"nvflare job {invoked_sub_cmd} abc123 concise",
        ],
        sys.argv[1:],
    )

    level = getattr(cmd_args, "level", None)
    site = getattr(cmd_args, "site", "all")

    if not level:
        output_usage_error(
            job_sub_cmd_parser[CMD_JOB_LOG_CONFIG],
            "provide a valid level name or mode",
            exit_code=4,
            error_code="LOG_CONFIG_INVALID",
            message="Log config is not a recognised log mode.",
            hint="Supply one of: DEBUG, INFO, WARNING, ERROR, CRITICAL, concise, msg_only, full, verbose, reload.",
        )
        return

    try:
        with _job_session_for_args(cmd_args) as sess:
            meta = sess.get_job_meta(cmd_args.job_id)
            job_status = meta.get("status", "UNKNOWN") if meta else "UNKNOWN"
            if job_status in _TERMINAL_JOB_STATES:
                output_error(
                    "JOB_NOT_RUNNING",
                    exit_code=1,
                    job_id=cmd_args.job_id,
                    detail=f"job is in terminal state: {job_status}",
                )
                raise SystemExit(1)
            sess.configure_job_log(cmd_args.job_id, level, target=site)
    except (AuthenticationError, AuthorizationError, NoConnection):
        raise
    except InternalError as e:
        output_error("INTERNAL_ERROR", exit_code=5, detail=str(e))
        return
    except InvalidTarget:
        output_error("SITE_NOT_FOUND", site=site)
        return
    except NoReply:
        output_error("SITE_NOT_FOUND", site=site)
        return
    except JobNotFound:
        output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        return
    except Exception as e:
        output_error("INTERNAL_ERROR", exit_code=5, detail=str(e))
        return

    sites = [site] if site != "all" else ["all"]
    output_ok({"job_id": cmd_args.job_id, "config": level, "sites": sites, "status": "applied"})


def define_job_monitor_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_MONITOR, help="wait for a job and stream progress to stderr")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument("--timeout", type=int, default=0, help="seconds to wait (0 = no timeout)")
    p.add_argument("--interval", type=int, default=2, help="poll interval in seconds")
    p.add_argument("--study", type=str, default="default", help="study to monitor the job in")
    p.add_argument(
        "--stats-target",
        dest="stats_target",
        choices=["server", "client", "all"],
        default="server",
        help="where to fetch stats from (default: server)",
    )
    p.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        default=None,
        help="extra metric key to surface from stats (repeatable)",
    )
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_MONITOR] = p
    job_sub_cmd_handlers[CMD_JOB_MONITOR] = cmd_job_monitor


def define_job_log_parser(job_subparser):
    p = job_subparser.add_parser(
        CMD_JOB_LOG_CONFIG,
        aliases=[CMD_JOB_LOG_ALIAS],
        help="change logging configuration for a running job",
    )
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument(
        "level",
        nargs="?",
        default=None,
        help="log level or mode: DEBUG, INFO, WARNING, ERROR, CRITICAL, concise, msg_only, full, verbose, reload",
    )
    p.add_argument("--site", default="all", help="target site name or all")
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_LOG_CONFIG] = p
    job_sub_cmd_parser[CMD_JOB_LOG_ALIAS] = p
    job_sub_cmd_handlers[CMD_JOB_LOG_CONFIG] = cmd_job_log
    job_sub_cmd_handlers[CMD_JOB_LOG_ALIAS] = cmd_job_log

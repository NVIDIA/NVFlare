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
import os
import shutil
import sys
import traceback
from tempfile import mkdtemp
from typing import List, Optional, Tuple


class _WiderSubcmdFormatter(argparse.HelpFormatter):
    """Formatter that prevents long subcommand names from wrapping to the next line.

    argparse computes _action_max_length at section-indent level (2) but renders at
    subsection-indent level (4), causing a 2-char gap that makes names like
    'list_templates' (14 chars) wrap.  Adding indent_increment to the computation
    closes the gap.
    """

    def add_arguments(self, actions):
        for action in actions:
            invocations = [self._format_action_invocation(action)]
            for subaction in self._iter_indented_subactions(action):
                invocations.append(self._format_action_invocation(subaction))
            invocation_length = max(map(len, invocations))
            self._action_max_length = max(
                self._action_max_length,
                invocation_length + self._current_indent + self._indent_increment,
            )
        super().add_arguments(actions)


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
CMD_JOB_NEW = "new"
CMD_JOB_LIST = "list"
CMD_JOB_META = "meta"
CMD_JOB_ABORT = "abort"
CMD_JOB_CLONE = "clone"
CMD_JOB_DOWNLOAD = "download"
CMD_JOB_DELETE = "delete"

# Section 5: Job observability commands
CMD_JOB_STATS = "stats"
CMD_JOB_ERRORS = "errors"
CMD_JOB_WAIT = "wait"
CMD_JOB_LOGS = "logs"
CMD_JOB_LOG_LEVEL = "log-level"
CMD_JOB_DIAGNOSE = "diagnose"

# Section 3 additions: new lifecycle commands
CMD_JOB_MONITOR = "monitor"
CMD_JOB_LOG = "log"
CMD_JOB_EXPORT = "export"
CMD_JOB_RUN = "run"


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
        deprecated_message="Use 'nvflare job new -r <recipe> --script <train.py>' instead.",
    )
    print(
        "WARNING: 'nvflare job create' is deprecated. Use 'nvflare job new -r <recipe> --script <train.py>' instead."
        " Run 'nvflare recipe list' to see available recipes.",
        file=sys.stderr,
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
        from nvflare.tool.cli_output import print_human

        print_human(f"\nUnable to handle command: {CMD_CREATE_JOB} due to: {e} \n")
        if cmd_args.debug:
            print_human(traceback.format_exc())
        sub_cmd_parser = job_sub_cmd_parser[CMD_CREATE_JOB]
        if sub_cmd_parser:
            sub_cmd_parser.print_help(sys.stderr)


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
        "nvflare job show_variables",
        [],
        sys.argv[1:],
        deprecated=True,
        deprecated_message="Use the Job Recipe API instead.",
    )
    print("WARNING: 'nvflare job show_variables' is deprecated. Use the Job Recipe API instead.", file=sys.stderr)
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
        from nvflare.tool.cli_output import print_human

        print_human(f"\nUnable to handle command: {CMD_SHOW_VARIABLES} due to: {e} \n")
        if cmd_args.debug:
            print_human(traceback.format_exc())
        sub_cmd_parser = job_sub_cmd_parser[CMD_SHOW_VARIABLES]
        if sub_cmd_parser:
            sub_cmd_parser.print_help(sys.stderr)


def check_template_exists(target_template_name, template_index_conf):
    targets = [os.path.basename(key) for key in template_index_conf.get("templates").keys()]
    found = target_template_name in targets

    if not found:
        raise ValueError(
            f"Invalid template name {target_template_name}, "
            f"please check the available templates using nvflare job list_templates"
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
        "nvflare job list_templates",
        [],
        sys.argv[1:],
        deprecated=True,
        deprecated_message="Use 'nvflare recipe list' instead.",
    )
    print("WARNING: 'nvflare job list_templates' is deprecated. Use 'nvflare recipe list' instead.", file=sys.stderr)
    try:
        job_templates_dir = find_job_templates_location(cmd_args.job_templates_dir)
        job_templates_dir = os.path.abspath(job_templates_dir)
        template_index_conf = build_job_template_indices(job_templates_dir)
        display_available_templates(template_index_conf)

        if job_templates_dir:
            update_job_templates_dir(job_templates_dir)

    except ValueError as e:
        from nvflare.tool.cli_output import print_human

        print_human(f"\nUnable to handle command: {CMD_LIST_TEMPLATES} due to: {e} \n")
        if cmd_args.debug:
            print_human(traceback.format_exc())
        sub_cmd_parser = job_sub_cmd_parser[CMD_LIST_TEMPLATES]
        if sub_cmd_parser:
            sub_cmd_parser.print_help(sys.stderr)


def update_job_templates_dir(job_templates_dir: str):
    config_file_path, nvflare_config = get_hidden_config()
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

    handle_schema_flag(
        job_sub_cmd_parser[CMD_SUBMIT_JOB],
        "nvflare job submit",
        ["nvflare job submit -j ./my_job"],
        sys.argv[1:],
    )

    temp_job_dir = None
    try:
        if not os.path.isdir(cmd_args.job_folder):
            raise ValueError(f"invalid job folder: {cmd_args.job_folder}")

        temp_job_dir = mkdtemp()
        shutil.copytree(cmd_args.job_folder, temp_job_dir, dirs_exist_ok=True)

        app_dirs = get_app_dirs_from_job_folder(cmd_args.job_folder)
        app_names = [os.path.basename(f) for f in app_dirs]
        app_names = app_names if app_names else [DEFAULT_APP_NAME]

        prepare_job_config(cmd_args, app_names, temp_job_dir)
        admin_username, admin_user_dir = find_admin_user_and_dir()
        internal_submit_job(admin_user_dir, admin_username, temp_job_dir, cmd_args)

    except ValueError as e:
        from nvflare.tool.cli_output import print_human

        print_human(f"\nUnable to handle command: {CMD_SUBMIT_JOB} due to: {e} \n")
        if cmd_args.debug:
            print_human(traceback.format_exc())
        sub_cmd_parser = job_sub_cmd_parser[CMD_SUBMIT_JOB]
        if sub_cmd_parser:
            sub_cmd_parser.print_help(sys.stderr)
    finally:
        if temp_job_dir:
            if cmd_args.debug:
                from nvflare.tool.cli_output import print_human

                print_human(f"in debug mode, job configurations can be examined in temp job directory '{temp_job_dir}'")
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


def internal_submit_job(admin_user_dir, username, temp_job_dir, cmd_args=None):
    from nvflare.tool.cli_output import output_error, output_ok, print_human

    print_human("trying to connect to the server")
    sess = new_secure_session(username=username, startup_kit_location=admin_user_dir)
    try:
        job_id = sess.submit_job(temp_job_dir)
    except Exception as e:
        output_error("JOB_INVALID", detail=str(e))

    wait = getattr(cmd_args, "wait", False) if cmd_args else False
    timeout = getattr(cmd_args, "timeout", 0) if cmd_args else 0

    if wait:
        try:
            meta = sess.monitor_job(job_id, timeout=timeout if timeout else 0)
            output_ok(meta if isinstance(meta, dict) else {"job_id": job_id, "status": str(meta)})
        except TimeoutError:
            output_error("TIMEOUT", exit_code=3)
    else:
        output_ok({"job_id": job_id})


job_sub_cmd_handlers = {
    CMD_CREATE_JOB: create_job,
    CMD_SUBMIT_JOB: submit_job,
    CMD_LIST_TEMPLATES: list_templates,
    CMD_SHOW_VARIABLES: show_variables,
    CMD_JOB_NEW: None,  # set after function definition
    CMD_JOB_LIST: None,
    CMD_JOB_META: None,
    CMD_JOB_ABORT: None,
    CMD_JOB_CLONE: None,
    CMD_JOB_DOWNLOAD: None,
    CMD_JOB_DELETE: None,
    CMD_JOB_STATS: None,
    CMD_JOB_ERRORS: None,
    CMD_JOB_WAIT: None,
    CMD_JOB_LOGS: None,
    CMD_JOB_LOG_LEVEL: None,
    CMD_JOB_DIAGNOSE: None,
    CMD_JOB_MONITOR: None,
    CMD_JOB_LOG: None,
    CMD_JOB_EXPORT: None,
    CMD_JOB_RUN: None,
}

job_sub_cmd_parser = {
    CMD_CREATE_JOB: None,
    CMD_SUBMIT_JOB: None,
    CMD_LIST_TEMPLATES: None,
    CMD_SHOW_VARIABLES: None,
    CMD_JOB_NEW: None,
    CMD_JOB_LIST: None,
    CMD_JOB_META: None,
    CMD_JOB_ABORT: None,
    CMD_JOB_CLONE: None,
    CMD_JOB_DOWNLOAD: None,
    CMD_JOB_DELETE: None,
    CMD_JOB_STATS: None,
    CMD_JOB_ERRORS: None,
    CMD_JOB_WAIT: None,
    CMD_JOB_LOGS: None,
    CMD_JOB_LOG_LEVEL: None,
    CMD_JOB_DIAGNOSE: None,
    CMD_JOB_MONITOR: None,
    CMD_JOB_LOG: None,
    CMD_JOB_EXPORT: None,
    CMD_JOB_RUN: None,
}


def handle_job_cli_cmd(cmd_args):
    job_cmd_handler = job_sub_cmd_handlers.get(cmd_args.job_sub_cmd, None)
    if job_cmd_handler:
        job_cmd_handler(cmd_args)
    elif cmd_args.job_sub_cmd is None:
        raise CLIUnknownCmdException("\n no job subcommand provided. \n")
    else:
        raise CLIUnknownCmdException("\n invalid command. \n")


def def_job_cli_parser(sub_cmd):
    cmd = "job"
    parser = sub_cmd.add_parser(cmd, help="submit, manage, and monitor FL jobs", formatter_class=_WiderSubcmdFormatter)
    job_subparser = parser.add_subparsers(title="job subcommands", metavar="", dest="job_sub_cmd")
    define_job_new_parser(job_subparser)
    define_submit_job_parser(job_subparser)
    define_job_monitor_parser(job_subparser)
    define_job_export_parser(job_subparser)
    define_job_run_parser(job_subparser)
    define_list_jobs_parser(job_subparser)
    define_abort_job_parser(job_subparser)
    define_job_meta_parser(job_subparser)
    define_job_logs_parser(job_subparser)
    define_job_log_parser(job_subparser)
    define_job_log_level_parser(job_subparser)
    define_job_wait_parser(job_subparser)
    define_job_stats_parser(job_subparser)
    define_job_errors_parser(job_subparser)
    define_job_diagnose_parser(job_subparser)
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
    submit_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    submit_parser.add_argument(
        "--wait", action="store_true", default=False, help="block until job reaches terminal state"
    )
    submit_parser.add_argument("--timeout", type=int, default=0, help="timeout in seconds for --wait (0 = no timeout)")
    submit_parser.add_argument("--study", type=str, default="default", help="study to submit the job to")
    job_sub_cmd_parser[CMD_SUBMIT_JOB] = submit_parser


def define_list_templates_parser(job_subparser):
    show_jobs_parser = job_subparser.add_parser("list_templates", help="[DEPRECATED] use 'nvflare recipe list'")
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
        "show_variables", help="[DEPRECATED] use 'nvflare recipe list' or the Job Recipe API"
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
    show_variables_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_SHOW_VARIABLES] = show_variables_parser


def define_create_job_parser(job_subparser):
    create_parser = job_subparser.add_parser(
        "create", help="[DEPRECATED] use 'nvflare job new -r <recipe> --script <train.py>'"
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


def _coerce(value: str):
    """Auto-coerce a string to int, float, bool, or str."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _get_session(admin_user_dir=None, username=None, study="default"):
    """Create a secure session using the startup kit."""
    if admin_user_dir is None or username is None:
        u, d = find_admin_user_and_dir()
        if username is None:
            username = u
        if admin_user_dir is None:
            admin_user_dir = d
    return new_secure_session(username=username, startup_kit_location=admin_user_dir)


def cmd_job_new(cmd_args):
    import importlib

    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_NEW],
        "nvflare job new",
        [
            "nvflare job new -r fedavg --script train.py",
            "nvflare job new -r fedavg --script train.py --min-clients 3 --param rounds=20",
        ],
        sys.argv[1:],
    )

    try:
        from nvflare.tool.recipe.recipe_cli import _load_catalog
    except ImportError:
        output_error(
            "INVALID_ARGS",
            exit_code=4,
            detail="recipe catalog not available",
        )
        return  # unreachable; output_error exits

    catalog = _load_catalog()

    if not cmd_args.recipe:
        output_ok({"available_recipes": [r["name"] for r in catalog], "hint": "re-run with -r <recipe> to scaffold"})
        return

    entry = next((r for r in catalog if r["name"] == cmd_args.recipe), None)
    if entry is None:
        output_error(
            "INVALID_ARGS",
            exit_code=4,
            detail=f"unknown recipe '{cmd_args.recipe}'. Available: {[r['name'] for r in catalog]}",
        )
        return  # unreachable; output_error exits

    params = {}
    for p in getattr(cmd_args, "param", []):
        k, _, v = p.partition("=")
        params[k.strip()] = _coerce(v.strip())

    job_folder = os.path.abspath(cmd_args.job_folder)

    try:
        module = importlib.import_module(entry["module"])
        RecipeClass = getattr(module, entry["class"])
        recipe = RecipeClass(script=cmd_args.script, **params)

        from nvflare.job_config.api import FedJob

        job = FedJob(name=os.path.basename(job_folder), min_clients=cmd_args.min_clients)
        job.to(recipe)
        job.export_job(os.path.dirname(job_folder) or ".")
    except ImportError as e:
        output_error("INVALID_ARGS", exit_code=4, detail=str(e))
        return
    except Exception as e:
        output_error("INTERNAL_ERROR", exit_code=5, detail=str(e))
        return

    output_ok(
        {
            "job_folder": job_folder,
            "recipe": cmd_args.recipe,
            "min_clients": cmd_args.min_clients,
            "script": cmd_args.script,
            "params": params,
        }
    )


def cmd_job_list(cmd_args):
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

    try:
        sess = _get_session()
        jobs = sess.list_jobs(
            name_prefix=getattr(cmd_args, "name", None),
            id_prefix=getattr(cmd_args, "id", None),
            reverse=getattr(cmd_args, "reverse", False),
            limit=getattr(cmd_args, "max", None),
        )
    except Exception as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    for j in jobs:
        if "study" not in j:
            j["study"] = getattr(cmd_args, "study", "default")

    output_ok(jobs)


def cmd_job_meta(cmd_args):
    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_META],
        "nvflare job meta",
        ["nvflare job meta <job_id>"],
        sys.argv[1:],
    )

    try:
        sess = _get_session()
        meta = sess.get_job_meta(cmd_args.job_id)
    except Exception as e:
        err = str(e).lower()
        if "not found" in err or "does not exist" in err:
            output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        else:
            output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    if meta is None:
        output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
    else:
        output_ok(meta)


def cmd_job_abort(cmd_args):
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
            return
        from nvflare.tool.cli_output import print_human, prompt_yn

        if not prompt_yn(f"Abort job '{cmd_args.job_id}'?"):
            print_human("Aborted.")
            return

    try:
        sess = _get_session()
        sess.abort_job(cmd_args.job_id)
    except Exception as e:
        err = str(e).lower()
        if "not found" in err or "does not exist" in err:
            output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        elif "not running" in err or "not active" in err:
            output_error("JOB_NOT_RUNNING", job_id=cmd_args.job_id)
        else:
            output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    output_ok({"job_id": cmd_args.job_id, "status": "ABORTED"})


def cmd_job_clone(cmd_args):
    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_CLONE],
        "nvflare job clone",
        ["nvflare job clone <job_id>"],
        sys.argv[1:],
    )

    try:
        sess = _get_session()
        new_job_id = sess.clone_job(cmd_args.job_id)
    except Exception as e:
        err = str(e).lower()
        if "not found" in err or "does not exist" in err:
            output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        else:
            output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    output_ok({"source_job_id": cmd_args.job_id, "new_job_id": new_job_id})


def cmd_job_download(cmd_args):
    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_DOWNLOAD],
        "nvflare job download",
        ["nvflare job download <job_id>", "nvflare job download <job_id> -o /path/to/results"],
        sys.argv[1:],
    )

    destination = os.path.abspath(getattr(cmd_args, "output_dir", "./"))
    try:
        sess = _get_session()
        path = sess.download_job_result(cmd_args.job_id, destination)
    except Exception as e:
        err = str(e).lower()
        if "not found" in err or "does not exist" in err:
            output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        else:
            output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    output_ok({"job_id": cmd_args.job_id, "path": path or destination})


def cmd_job_delete(cmd_args):
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
            return
        from nvflare.tool.cli_output import print_human, prompt_yn

        if not prompt_yn(f"Delete job '{cmd_args.job_id}'?"):
            print_human("Cancelled.")
            return

    try:
        sess = _get_session()
        sess.delete_job(cmd_args.job_id)
    except Exception as e:
        err = str(e).lower()
        if "not found" in err or "does not exist" in err:
            output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        else:
            output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    output_ok({"job_id": cmd_args.job_id})


# ---------------------------------------------------------------------------
# Parser definitions for new commands
# ---------------------------------------------------------------------------


def define_job_new_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_NEW, help="scaffold a new job from a recipe")
    p.add_argument("-r", "--recipe", type=str, default=None, help="recipe name; omit to list available recipes")
    p.add_argument("--script", type=str, required=True, help="path to the training script")
    p.add_argument("-j", "--job-folder", dest="job_folder", type=str, default="./current_job", help="output job folder")
    p.add_argument("--script-dir", dest="script_dir", type=str, default=None, help="additional files directory")
    p.add_argument("--min-clients", dest="min_clients", type=int, default=2, help="minimum number of FL clients")
    p.add_argument("--study", type=str, default="default", help="study this job belongs to")
    p.add_argument(
        "--param", type=str, action="append", default=[], metavar="key=value", help="recipe parameter (repeatable)"
    )
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_NEW] = p
    job_sub_cmd_handlers[CMD_JOB_NEW] = cmd_job_new


def define_list_jobs_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_LIST, help="list jobs on the server")
    p.add_argument("-n", "--name", type=str, default=None, help="filter by name prefix")
    p.add_argument("-i", "--id", type=str, default=None, help="filter by job ID prefix")
    p.add_argument("-r", "--reverse", action="store_true", default=False, help="reverse sort order")
    p.add_argument("-m", "--max", type=int, default=None, help="max results to return")
    p.add_argument("--study", type=str, default="default", help="study to list jobs from; use 'all' for all studies")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_LIST] = p
    job_sub_cmd_handlers[CMD_JOB_LIST] = cmd_job_list


def define_job_meta_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_META, help="get metadata for a job")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_META] = p
    job_sub_cmd_handlers[CMD_JOB_META] = cmd_job_meta


def define_abort_job_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_ABORT, help="abort a running job")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument("--force", action="store_true", help="skip confirmation prompt")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_ABORT] = p
    job_sub_cmd_handlers[CMD_JOB_ABORT] = cmd_job_abort


def define_clone_job_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_CLONE, help="clone an existing job")
    p.add_argument("job_id", type=str, help="job ID to clone")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_CLONE] = p
    job_sub_cmd_handlers[CMD_JOB_CLONE] = cmd_job_clone


def define_download_job_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_DOWNLOAD, help="download job result")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument("-o", "--output-dir", dest="output_dir", type=str, default="./", help="destination directory")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_DOWNLOAD] = p
    job_sub_cmd_handlers[CMD_JOB_DOWNLOAD] = cmd_job_download


def define_delete_job_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_DELETE, help="delete a job")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument("--force", action="store_true", help="skip confirmation prompt")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_DELETE] = p
    job_sub_cmd_handlers[CMD_JOB_DELETE] = cmd_job_delete


# ---------------------------------------------------------------------------
# Section 5: Job observability commands
# ---------------------------------------------------------------------------

_DIAGNOSE_PATTERNS = [
    (r"CUDA out of memory|OOM", "GPU memory exhaustion", "Reduce batch_size or enable gradient checkpointing"),
    (r"Connection refused|SERVER_UNREACHABLE", "Network / firewall issue", "Check server status and network policy"),
    (r"signature verification failed", "Certificate mismatch", "Re-provision or verify rootCA"),
    (r"Job validation failed", "Bad job configuration", "Check meta.json and config_fed_server.json"),
    (r"timed out", "Client too slow", "Increase task_timeout in job config"),
    (r"ModuleNotFoundError", "Missing dependency", "Install required package on the client"),
    (r"ResourceExhaustedError", "Server memory pressure", "Reduce concurrent jobs or client batch size"),
    (r"SSLError|certificate verify failed", "TLS misconfiguration", "Check cert expiry and rootCA chain"),
]

_TERMINAL_JOB_STATES = {"FINISHED_OK", "FINISHED_EXCEPTION", "ABORTED", "ABANDONED", "FAILED"}


def cmd_job_stats(cmd_args):
    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_STATS],
        "nvflare job stats",
        ["nvflare job stats abc123", "nvflare job stats abc123 --site server"],
        sys.argv[1:],
    )

    try:
        sess = _get_session()
        result = sess.show_stats(cmd_args.job_id, "all", None)
    except Exception as e:
        err = str(e).lower()
        if "not found" in err or "does not exist" in err:
            output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        else:
            output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    output_ok({"job_id": cmd_args.job_id, "stats": result})


def cmd_job_errors(cmd_args):
    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_ERRORS],
        "nvflare job errors",
        ["nvflare job errors abc123"],
        sys.argv[1:],
    )

    try:
        sess = _get_session()
        result = sess.show_errors(cmd_args.job_id, "all", None)
    except Exception as e:
        err = str(e).lower()
        if "not found" in err or "does not exist" in err:
            output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        else:
            output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    output_ok({"job_id": cmd_args.job_id, "errors": result})


def cmd_job_wait(cmd_args):
    import time

    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_WAIT],
        "nvflare job wait",
        ["nvflare job wait abc123 --timeout 300"],
        sys.argv[1:],
    )

    start = time.time()
    try:
        sess = _get_session()
        meta = sess.wait_for_job(cmd_args.job_id, timeout=cmd_args.timeout, poll_interval=cmd_args.interval)
    except TimeoutError:
        output_error("TIMEOUT", exit_code=3, detail="job did not reach terminal state within timeout")
        return
    except Exception as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    status = meta.get("status", "UNKNOWN")
    duration = round(time.time() - start, 1)
    data = {"job_id": cmd_args.job_id, "status": status, "duration_s": duration}

    if status in ("FAILED", "ABORTED", "FINISHED_EXCEPTION", "ABANDONED"):
        output_ok(data)
        sys.exit(1)

    output_ok(data)


def cmd_job_logs(cmd_args):
    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_LOGS],
        "nvflare job logs",
        ["nvflare job logs abc123", "nvflare job logs abc123 --tail 100 --site server"],
        sys.argv[1:],
    )

    try:
        sess = _get_session()
        result = sess.get_job_logs(
            cmd_args.job_id,
            target=cmd_args.site,
            tail_lines=cmd_args.tail,
            grep_pattern=cmd_args.grep,
        )
    except Exception as e:
        err = str(e).lower()
        if "not found" in err or "does not exist" in err:
            output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        else:
            output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    output_ok(
        {
            "job_id": cmd_args.job_id,
            "log_source": result.get("log_source", "workspace"),
            "logs": result.get("logs", {}),
        }
    )


def cmd_job_log_level(cmd_args):
    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag
    from nvflare.tool.system.system_cli import resolve_log_config

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_LOG_LEVEL],
        "nvflare job log-level",
        ["nvflare job log-level abc123 DEBUG", "nvflare job log-level abc123 --config /path/to/logging.json"],
        sys.argv[1:],
    )

    level = getattr(cmd_args, "level", None)
    config_str = getattr(cmd_args, "config", None)
    site = getattr(cmd_args, "site", "all")

    log_config = resolve_log_config(level, config_str)
    if log_config is None:
        output_error("LOG_CONFIG_INVALID", detail="provide a valid level name or --config JSON/file")
        return

    try:
        sess = _get_session()
        # Check job state first
        meta = sess.get_job_meta(cmd_args.job_id)
        job_status = meta.get("status", "UNKNOWN") if meta else "UNKNOWN"
        if job_status in _TERMINAL_JOB_STATES:
            output_error(
                "JOB_NOT_RUNNING",
                exit_code=1,
                detail=f"job is in terminal state: {job_status}",
            )
            return
        sess.configure_job_log(cmd_args.job_id, log_config, target=site)
    except Exception as e:
        from nvflare.fuel.flare_api.api_spec import NoConnection

        if isinstance(e, NoConnection):
            output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        elif "not found" in str(e).lower() or "does not exist" in str(e).lower():
            output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        else:
            output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    output_ok({"job_id": cmd_args.job_id, "site": site, "log_config": log_config, "status": "applied"})


def cmd_job_diagnose(cmd_args):
    import re

    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_DIAGNOSE],
        "nvflare job diagnose",
        ["nvflare job diagnose abc123", "nvflare job diagnose abc123 --site server"],
        sys.argv[1:],
    )

    try:
        sess = _get_session()
        meta = sess.get_job_meta(cmd_args.job_id)
        logs_result = sess.get_job_logs(cmd_args.job_id, target=cmd_args.site)
        errors_result = sess.show_errors(cmd_args.job_id, "all", None)
    except Exception as e:
        err = str(e).lower()
        if "not found" in err or "does not exist" in err:
            output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        else:
            output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    job_status = meta.get("status", "UNKNOWN") if meta else "UNKNOWN"
    logs = logs_result.get("logs", {})
    findings = []
    unexplained = []

    for site, log_text in logs.items():
        matched = False
        for pattern, diagnosis, action in _DIAGNOSE_PATTERNS:
            m = re.search(pattern, log_text, re.IGNORECASE)
            if m:
                excerpt_lines = [ln for ln in log_text.splitlines() if re.search(pattern, ln, re.IGNORECASE)]
                excerpt = "\n".join(excerpt_lines[:5])
                findings.append(
                    {
                        "site": site,
                        "pattern": m.group(0),
                        "diagnosis": diagnosis,
                        "action": action,
                        "log_excerpt": excerpt,
                    }
                )
                matched = True
                break
        if not matched and log_text.strip():
            lines = log_text.splitlines()
            unexplained.append({"site": site, "log_excerpt": "\n".join(lines[:20])})

    report_lines = [f"## Job Diagnosis: {cmd_args.job_id}\n\n**Status:** {job_status}\n"]
    if findings:
        for f in findings:
            report_lines.append(
                f"### {f['site']} — {f['diagnosis']}\n- **Action:** {f['action']}\n- **Excerpt:** `{f['log_excerpt'][:200]}`\n"
            )
    else:
        report_lines.append("No known failure patterns detected.\n")
    report = "\n".join(report_lines)

    data = {
        "job_id": cmd_args.job_id,
        "job_status": job_status,
        "findings": findings,
        "unexplained": unexplained,
        "report": report,
    }

    output_ok(data)


def define_job_stats_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_STATS, help="show running job statistics")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument("--site", default="all", help="target site name or all")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_STATS] = p
    job_sub_cmd_handlers[CMD_JOB_STATS] = cmd_job_stats


def define_job_errors_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_ERRORS, help="show job errors per site")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument("--site", default="all", help="target site name or all")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_ERRORS] = p
    job_sub_cmd_handlers[CMD_JOB_ERRORS] = cmd_job_errors


def define_job_wait_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_WAIT, help="wait for a job to reach terminal state")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument("--timeout", type=int, default=0, help="seconds to wait (0 = no timeout)")
    p.add_argument("--interval", type=int, default=2, help="poll interval in seconds")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_WAIT] = p
    job_sub_cmd_handlers[CMD_JOB_WAIT] = cmd_job_wait


def define_job_logs_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_LOGS, help="retrieve job logs from server workspace")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument("--site", default="all", help="target site name or all")
    p.add_argument("--tail", type=int, default=None, help="number of tail lines to retrieve")
    p.add_argument("--grep", default=None, help="grep pattern to filter log lines")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_LOGS] = p
    job_sub_cmd_handlers[CMD_JOB_LOGS] = cmd_job_logs


def define_job_log_level_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_LOG_LEVEL, help="change logging level for a running job")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument("level", nargs="?", default=None, help="log level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    p.add_argument("--config", default=None, help="path to dictConfig JSON file or inline JSON")
    p.add_argument("--site", default="all", help="target site name or all")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_LOG_LEVEL] = p
    job_sub_cmd_handlers[CMD_JOB_LOG_LEVEL] = cmd_job_log_level


def define_job_diagnose_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_DIAGNOSE, help="analyze job logs and errors for known failure patterns")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument("--site", default="all", help="target site name or all")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_DIAGNOSE] = p
    job_sub_cmd_handlers[CMD_JOB_DIAGNOSE] = cmd_job_diagnose


# ---------------------------------------------------------------------------
# Section 3 additions: monitor, log, export, run
# ---------------------------------------------------------------------------


def cmd_job_monitor(cmd_args):
    import time

    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_MONITOR],
        "nvflare job monitor",
        ["nvflare job monitor abc123", "nvflare job monitor abc123 --timeout 3600"],
        sys.argv[1:],
    )

    start = time.time()
    timeout = getattr(cmd_args, "timeout", 0)
    interval = getattr(cmd_args, "interval", 2)

    try:
        sess = _get_session()
        meta = sess.wait_for_job(cmd_args.job_id, timeout=timeout, poll_interval=interval)
    except TimeoutError:
        output_error("TIMEOUT", exit_code=3, detail="job did not reach terminal state within timeout")
        return
    except Exception as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    status = meta.get("status", "UNKNOWN")
    duration = round(time.time() - start, 1)
    data = {"job_id": cmd_args.job_id, "status": status, "duration_s": duration}

    if status in ("FAILED", "FINISHED_EXCEPTION"):
        output_ok(data)
        sys.exit(1)
    elif status in ("ABORTED", "ABANDONED"):
        output_ok(data)
        sys.exit(1)

    output_ok(data)


def cmd_job_log(cmd_args):
    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag
    from nvflare.tool.system.system_cli import resolve_log_config

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_LOG],
        "nvflare job log",
        ["nvflare job log abc123 DEBUG", "nvflare job log abc123 --config /path/to/logging.json"],
        sys.argv[1:],
    )

    level = getattr(cmd_args, "level", None)
    config_str = getattr(cmd_args, "config", None)
    site = getattr(cmd_args, "site", "all")

    log_config = resolve_log_config(level, config_str)
    if log_config is None:
        output_error("LOG_CONFIG_INVALID", detail="provide a valid level name or --config JSON/file")
        return

    try:
        sess = _get_session()
        meta = sess.get_job_meta(cmd_args.job_id)
        job_status = meta.get("status", "UNKNOWN") if meta else "UNKNOWN"
        if job_status in _TERMINAL_JOB_STATES:
            output_error(
                "JOB_NOT_RUNNING",
                exit_code=1,
                detail=f"job is in terminal state: {job_status}",
            )
            return
        sess.configure_job_log(cmd_args.job_id, log_config, target=site)
    except Exception as e:
        from nvflare.fuel.flare_api.api_spec import NoConnection

        if isinstance(e, NoConnection):
            output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        elif "not found" in str(e).lower() or "does not exist" in str(e).lower():
            output_error("JOB_NOT_FOUND", job_id=cmd_args.job_id)
        else:
            output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return

    output_ok({"job_id": cmd_args.job_id, "site": site, "log_config": log_config, "status": "applied"})


def _resolve_and_export_recipe(recipe_folder: str, out_dir: str, entry=None):
    """Resolve a Recipe subclass and export it to *out_dir*.

    If *entry* is given (``"module:ClassName"``), load it directly.
    Otherwise scan *.py files in *recipe_folder* for a unique Recipe subclass.

    Calls ``output_error`` and returns ``False`` on any failure; returns ``True``
    on success.  Callers should ``return`` immediately when ``False`` is returned.
    """
    import glob as _glob
    import importlib
    import importlib.util
    import inspect

    from nvflare.tool.cli_output import output_error

    try:
        from nvflare.recipe.spec import Recipe

        if entry:
            module_name, _, class_name = entry.partition(":")
            mod = importlib.import_module(module_name)
            RecipeClass = getattr(mod, class_name)
        else:
            found = []
            for py_file in _glob.glob(os.path.join(recipe_folder, "*.py")):
                spec = importlib.util.spec_from_file_location("_recipe_scan", py_file)
                mod = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(mod)
                except Exception:
                    continue
                for name, obj in inspect.getmembers(mod, inspect.isclass):
                    if issubclass(obj, Recipe) and obj is not Recipe:
                        found.append((name, obj))
            if not found:
                output_error("RECIPE_ENTRY_NOT_FOUND")
                return False
            if len(found) > 1:
                output_error("RECIPE_ENTRY_AMBIGUOUS")
                return False
            _, RecipeClass = found[0]

        recipe_instance = RecipeClass()
        recipe_instance.export(out_dir)
    except (ImportError, AttributeError) as e:
        output_error("RECIPE_ENTRY_NOT_FOUND", detail=str(e))
        return False
    except Exception as e:
        output_error("RECIPE_EXPORT_FAILED", detail=str(e))
        return False

    return True


def cmd_job_export(cmd_args):
    from nvflare.tool.cli_output import output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_EXPORT],
        "nvflare job export",
        [
            "nvflare job export --recipe-folder . --out ./fl_job",
            "nvflare job export --recipe-folder . --entry recipe:MyRecipe --out ./fl_job",
        ],
        sys.argv[1:],
    )

    recipe_folder = os.path.abspath(cmd_args.recipe_folder)
    out_dir = os.path.abspath(cmd_args.out)

    if not _resolve_and_export_recipe(recipe_folder, out_dir, entry=getattr(cmd_args, "entry", None)):
        return

    output_ok({"job_folder": out_dir, "recipe_folder": recipe_folder})


def cmd_job_run(cmd_args):
    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        job_sub_cmd_parser[CMD_JOB_RUN],
        "nvflare job run",
        [
            "nvflare job run --recipe-folder . --env poc",
            "nvflare job run --recipe-folder . --env sim",
        ],
        sys.argv[1:],
    )

    import tempfile

    recipe_folder = os.path.abspath(cmd_args.recipe_folder)
    env = getattr(cmd_args, "env", "poc")

    with tempfile.TemporaryDirectory() as tmp_out:
        if not _resolve_and_export_recipe(recipe_folder, tmp_out, entry=getattr(cmd_args, "entry", None)):
            return

        # Step 2: submit or simulate
        if env == "sim":
            try:
                from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner

                runner = SimulatorRunner(job_folder=tmp_out, workspace=getattr(cmd_args, "workspace", "/tmp/sim_ws"))
                runner.run()
                output_ok({"status": "FINISHED_OK", "env": env, "job_folder": tmp_out})
            except Exception as e:
                output_error("INTERNAL_ERROR", exit_code=5, detail=str(e))
        else:
            try:
                sess = _get_session()
                job_id = sess.submit_job(tmp_out)
                output_ok({"job_id": job_id, "env": env})
            except Exception as e:
                output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))


def define_job_monitor_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_MONITOR, help="wait for a job and stream progress to stderr")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument("--timeout", type=int, default=0, help="seconds to wait (0 = no timeout)")
    p.add_argument("--interval", type=int, default=2, help="poll interval in seconds")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_MONITOR] = p
    job_sub_cmd_handlers[CMD_JOB_MONITOR] = cmd_job_monitor


def define_job_log_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_LOG, help="change logging configuration for a running job")
    p.add_argument("job_id", type=str, help="job ID")
    p.add_argument("level", nargs="?", default=None, help="log level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    p.add_argument("--config", default=None, help="path to dictConfig JSON file or inline JSON")
    p.add_argument("--site", default="all", help="target site name or all")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_LOG] = p
    job_sub_cmd_handlers[CMD_JOB_LOG] = cmd_job_log


def define_job_export_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_EXPORT, help="export a recipe folder to a job config folder")
    p.add_argument("--recipe-folder", dest="recipe_folder", type=str, default=".", help="folder containing recipe.py")
    p.add_argument("--out", type=str, required=True, help="output job folder path")
    p.add_argument("--entry", type=str, default=None, help="module:ClassName of the Recipe subclass")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_EXPORT] = p
    job_sub_cmd_handlers[CMD_JOB_EXPORT] = cmd_job_export


def define_job_run_parser(job_subparser):
    p = job_subparser.add_parser(CMD_JOB_RUN, help="export recipe and submit/simulate in one step")
    p.add_argument("--recipe-folder", dest="recipe_folder", type=str, default=".", help="folder containing recipe.py")
    p.add_argument(
        "--env",
        type=str,
        default="poc",
        choices=["sim", "poc", "prod"],
        help="execution environment: sim (simulator), poc, or prod",
    )
    p.add_argument(
        "--entry",
        type=str,
        default=None,
        help="module:ClassName of the Recipe subclass. If omitted, auto-searches --recipe-folder for a Recipe subclass.",
    )
    p.add_argument("--workspace", type=str, default="/tmp/sim_ws", help="workspace directory for the job run.")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    job_sub_cmd_parser[CMD_JOB_RUN] = p
    job_sub_cmd_handlers[CMD_JOB_RUN] = cmd_job_run

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
import json
import os
import pathlib
import shutil
from distutils.dir_util import copy_tree
from tempfile import mkdtemp
from typing import List, Optional, Tuple

from pyhocon import ConfigFactory as CF
from pyhocon import ConfigTree

from nvflare.apis.job_def import JobMetaKey
from nvflare.cli_exception import CLIException
from nvflare.fuel.flare_api.flare_api import new_secure_session
from nvflare.fuel.utils.config_factory import ConfigFactory
from nvflare.tool.job.config.configer import merge_configs_from_cli
from nvflare.utils.cli_utils import get_startup_kit_dir, get_curr_dir, save_config, append_if_not_in_list

CMD_CREATE_JOB = "create"
CMD_SUBMIT_JOB = "submit"


def create_job(cmd_args):
    prepare_job_folder(cmd_args)
    predefined = load_predefined_config()
    prepare_fed_config(cmd_args, predefined)
    prepare_meta_config(cmd_args)
    prepare_model_exchange_config(cmd_args, predefined)


def submit_job(cmd_args):
    temp_job_dir = None
    try:
        temp_job_dir = mkdtemp()
        copy_tree(cmd_args.job_folder, temp_job_dir)

        prepare_submit_job_config(cmd_args, temp_job_dir)
        admin_username, admin_user_dir = find_admin_user_and_dir()
        internal_submit_job(admin_user_dir, admin_username, temp_job_dir)
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

    return admin_username, admin_user_dir


def internal_submit_job(admin_user_dir, username, temp_job_dir):
    sess = new_secure_session(username=username, startup_kit_location=admin_user_dir)
    job_id = sess.submit_job(temp_job_dir)
    print(f"job: '{job_id} was submitted")


job_sub_cmd_handlers = {
    CMD_CREATE_JOB: create_job,
    CMD_SUBMIT_JOB: submit_job
}


def handle_job_cli_cmd(cmd_args):
    job_cmd_handler = job_sub_cmd_handlers.get(cmd_args.job_sub_cmd, None)
    job_cmd_handler(cmd_args)


def def_job_cli_parser(sub_cmd):
    cmd = "job"
    parser = sub_cmd.add_parser(cmd)
    job_subparser = parser.add_subparsers(title="job", dest="job_sub_cmd", help="job subcommand")
    define_create_job_parser(job_subparser)
    define_submit_job_parser(job_subparser)

    return {cmd: parser}


def define_submit_job_parser(job_subparser):
    submit_parser = job_subparser.add_parser("submit",
                                             help="submit job")
    submit_parser.add_argument("-j", "--job_folder",
                               type=str,
                               nargs="?",
                               default=get_curr_dir(),
                               help="job_folder path, default to current directory")
    submit_parser.add_argument("-f", "--config_file",
                               type=str,
                               action='append',
                               nargs="*",
                               help="""Training config file with corresponding optional key=value pairs. 
                                       If key presents in the preceding config file, the value in the config
                                       file will be overwritten by the new value """)
    submit_parser.add_argument("-a", "--app_config",
                               type=str,
                               nargs="*",
                               help="""key=value options will be passed directly to script argument """)

    submit_parser.add_argument("-debug", "--debug", action='store_true', help="debug is on")


def define_create_job_parser(job_subparser):
    create_parser = job_subparser.add_parser("create", help="create job")
    create_parser.add_argument("-j", "--job_folder",
                               type=str,
                               nargs="?",
                               default=get_curr_dir(),
                               help="job_folder path, default to current directory")
    create_parser.add_argument("-w", "--workflows",
                               type=str,
                               nargs="*",
                               default=["SAG"],
                               help="""Workflows available works are
                                       SAG for ScatterAndGather,
                                       CROSS for CrossSiteModelEval or
                                       CYCLIC for CyclicController """)
    create_parser.add_argument("-m", "--min_clients",
                               type=int, nargs="?",
                               default=1,
                               help="min clients default to 1")
    create_parser.add_argument("-n", "--num_rounds",
                               type=int,
                               nargs="?",
                               default=1,
                               help="number of total rounds, default to 1'")
    create_parser.add_argument("-s", "--script",
                               type=str,
                               nargs="?",
                               help="""code script such as train.py""")
    create_parser.add_argument("-sd", "--script_dir",
                               type=str,
                               nargs="?",
                               help="""script directory contains additional related files. 
                                       All files or directories under this directory will be copied over 
                                       to the custom directory.""")
    create_parser.add_argument("-debug", "--debug", action='store_true', help="debug is on")
    create_parser.add_argument("-force", "--force",
                               action='store_true',
                               help="force create is on, if -force, "
                                    "overwrite existing configuration with newly created configurations")


# ====================================================================
def prepare_submit_job_config(cmd_args, tmp_job_dir):
    update_client_app_script(cmd_args)
    merged_conf = merge_configs_from_cli(cmd_args)
    save_merged_configs(merged_conf, tmp_job_dir)


def update_client_app_script(cmd_args):
    if cmd_args.app_config:
        script_args = " ".join([f"--{k}" for k in cmd_args.app_config])
        config = ConfigFactory.load_config("config_fed_client.xxx")
        client_config = CF.from_dict(config.to_dict())
        client_config.put("app_script", script_args)
        save_config(client_config, config.file_path)


def save_merged_configs(merged_conf, tmp_job_dir):
    for file, file_configs in merged_conf.items():
        config_dir = pathlib.Path(tmp_job_dir) / "app" / "config"
        base_filename = os.path.basename(file)
        if base_filename.startswith("meta."):
            config_dir = tmp_job_dir
        base_filename = os.path.splitext(base_filename)[0]
        dst_path = config_dir / f"{base_filename}.json"
        save_config(file_configs, dst_path)


def get_upload_dir(startup_dir) -> str:
    console_config_path = os.path.join(startup_dir, "fed_admin.json")
    try:
        with open(console_config_path, "r") as f:
            console_config = json.load(f)
            upload_dir = console_config["admin"]["upload_dir"]
    except IOError as e:
        raise CLIException(f"failed to load {console_config_path} {e}")
    except json.decoder.JSONDecodeError as e:
        raise CLIException(f"failed to load {console_config_path}, please double check the configuration {e}")
    return upload_dir


def prepare_model_exchange_config(cmd_args, predefined):
    dst_path = dst_config_path(cmd_args, "config_exchange.json")
    if os.path.isfile(dst_path) and not cmd_args.force:
        return

    dst_config = load_src_config_template("config_exchange.conf")
    save_config(dst_config, dst_path)


def load_predefined_config():
    file_dir = os.path.dirname(__file__)
    return CF.parse_file(os.path.join(file_dir, "config/pre_defined.conf"))


def prepare_meta_config(cmd_args):
    job_folder = cmd_args.job_folder
    app_name = os.path.basename(job_folder)
    dst_path = os.path.join(job_folder, "meta.json")
    if os.path.isfile(dst_path) and not cmd_args.force:
        return
    dst_config = load_src_config_template("meta.conf")
    dst_config.put("name", app_name)
    dst_config.put(JobMetaKey.MIN_CLIENTS, cmd_args.min_clients)
    save_config(dst_config, dst_path)


def load_src_config_template(config_file_name: str):
    file_dir = os.path.dirname(__file__)
    config_template = CF.parse_file(os.path.join(file_dir, f"config/{config_file_name}"))
    return config_template


def prepare_fed_config(cmd_args, predefined):
    server_dst_path = dst_config_path(cmd_args, "config_fed_server.json")
    client_dst_path = dst_config_path(cmd_args, "config_fed_client.json")

    if (os.path.isfile(server_dst_path) or os.path.isfile(client_dst_path)) and not cmd_args.force:
        print(f"""warning: configuration files:
                {server_dst_path} 
                {client_dst_path} 
                already exists. Not generating the config files. If you would like to overwrite, use -force option""")
        return

    server_config, client_config = prepare_workflows(cmd_args, predefined)
    save_config(server_config, server_dst_path)
    save_config(client_config, client_dst_path)


def dst_app_path(cmd_args):
    job_folder = cmd_args.job_folder
    return os.path.join(job_folder, "app")


def dst_config_path(cmd_args, config_filename):
    app_dir = dst_app_path(cmd_args)
    config_dir = os.path.join(app_dir, "config")
    dst_path = os.path.join(config_dir, config_filename)
    return dst_path


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
                kv_dict[key] = value
            except ValueError:
                raise ValueError(f"Invalid key-value pair: '{kv}'")

    return kv_dict


def prepare_workflows(cmd_args, predefined) -> Tuple[ConfigTree, ConfigTree]:
    workflow_names: List[str] = cmd_args.workflows

    workflows_conf = predefined.get_config("workflows")
    invalid_names = [name for name in workflow_names if workflows_conf.get(name, None) is None]
    if invalid_names:
        raise ValueError(f"Unknown workflow names: {invalid_names}")

    server_config = CF.parse_string("""{ format_version = 2 }""")
    client_config = CF.parse_string("""{ format_version = 2 }""")

    workflows = []
    wf_components = []
    wf_task_result_filters = []
    wf_task_task_data_filters = []

    executors = []
    exec_components = []
    exec_task_result_filters = []
    exec_task_data_filters = []

    for wf_name in workflow_names:
        target_wf_conf = workflows_conf.get(f"{wf_name}.workflow")

        # special case
        if cmd_args.min_clients and target_wf_conf.get("args.min_clients", None) is not None:
            target_wf_conf.put("args.min_clients", cmd_args.min_clients)
        if cmd_args.num_rounds and target_wf_conf.get("args.num_rounds", None) is not None:
            target_wf_conf.put("args.num_rounds", cmd_args.num_rounds)

        append_if_not_in_list(workflows, target_wf_conf)
        predefined_wf_components = workflows_conf.get(f"{wf_name}.components", None)
        predefined_wf_task_data_filters = workflows_conf.get(f"{wf_name}.task_data_filters", None)
        predefined_wf_task_result_filters = workflows_conf.get(f"{wf_name}.task_result_filters", None)

        if predefined_wf_task_data_filters:
            for name, data_filter in predefined_wf_task_data_filters.items():
                append_if_not_in_list(wf_task_task_data_filters, data_filter)

        if predefined_wf_task_result_filters:
            for name, result_filter in predefined_wf_task_result_filters.items():
                append_if_not_in_list(wf_task_result_filters, result_filter)


        if predefined_wf_components:
            for name, comp in predefined_wf_components.items():
                append_if_not_in_list(wf_components, comp)

        predefined_executors = workflows_conf.get(f"{wf_name}.executors", None)
        if predefined_executors:
            item_lens = len(predefined_executors.items())
            target_exec = None
            if item_lens == 0:
                target_exec = None
            elif item_lens == 1:
                name = next(iter(predefined_executors))
                target_exec = predefined_executors.get(name)
            else:  # > 1
                target_exec_list = [exec_conf for name, exec_conf in predefined_executors.items() if
                                    exec_conf.get("default", False) is True]
                if target_exec_list:
                    target_exec = target_exec_list[0]

            if target_exec:
                # target_exec.put("script", f"{os.path.basename(cmd_args.script)}")
                append_if_not_in_list(executors, target_exec.get("executor"))

                if target_exec.get("task_data_filters", None):
                    for name, data_filter in target_exec.get("task_data_filters").items():
                        append_if_not_in_list(exec_task_data_filters, data_filter)
                if target_exec.get("task_result_filters", None):
                    for name, result_filter in target_exec.get("task_result_filters").items():
                        append_if_not_in_list(exec_task_result_filters, result_filter)
                for name, comp in target_exec.get("components").items():
                    append_if_not_in_list(exec_components, comp)

    server_config.put("workflows", workflows)
    server_config.put("components", wf_components)
    server_config.put("task_data_filters", wf_task_task_data_filters)
    server_config.put("task_result_filters", wf_task_result_filters)

    client_config.put("script", "")
    client_config.put("app_config", f" ")

    if cmd_args.script:
        script = os.path.basename(cmd_args.script.split(' ')[0])
        client_config.put("script", f"{script}")

    client_config.put("executors", executors)
    client_config.put("components", exec_components)
    client_config.put("task_data_filters", exec_task_data_filters)
    client_config.put("task_result_filters", exec_task_result_filters)

    return server_config, client_config


def prepare_job_folder(cmd_args):
    job_folder = cmd_args.job_folder
    if job_folder:
        if not os.path.exists(job_folder):
            os.makedirs(job_folder)
        elif not os.path.isdir(job_folder):
            raise ValueError(f"job_folder '{job_folder}' exits but not directory")

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

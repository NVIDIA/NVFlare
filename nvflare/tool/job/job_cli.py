# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from pyhocon.converter import HOCONConverter

from nvflare.apis.job_def import JobMetaKey
from nvflare.cli_exception import CLIException
from nvflare.fuel.flare_api.flare_api import new_secure_session
from nvflare.fuel.utils.config_factory import ConfigFactory
from nvflare.lighter.service_constants import FlareServiceConstants
from nvflare.tool.job.config.configer import merge_configs_from_cli

CMD_CREATE_JOB = "create"
CMD_SUBMIT_JOB = "submit"


def get_startup_kit_dir(startup_kit_dir: Optional[str] = None) -> str:
    if startup_kit_dir:
        return startup_kit_dir

    # load from config file:
    startup_kit_dir = find_startup_kit_location()
    if startup_kit_dir is None:
        startup_kit_dir = os.getenv("NVFLARE_STARTUP_KIT_DIR")

    if startup_kit_dir is None or len(startup_kit_dir.strip()) == 0:
        raise ValueError("startup kit directory is not specified")
    else:
        return startup_kit_dir


def get_curr_dir():
    return os.path.curdir


def get_home_dir():
    from pathlib import Path
    return Path.home()


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
    create_parser.add_argument("-ep", "--enable_persistor", action='store_true', help="enable persistor is true")
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


def find_startup_kit_location():
    hidden_nvflare_config_file = get_hidden_nvflare_config_path()
    job_info_conf = load_job_info_config(hidden_nvflare_config_file)
    return job_info_conf.get_string("startup_kit.path", None)


def save_job_info(cmd_args):
    hidden_nvflare_config_file = get_hidden_nvflare_config_path()
    nvflare_config = CF.parse_string("{}")
    nvflare_config = create_job_info_config(cmd_args, nvflare_config)
    save_config(nvflare_config, hidden_nvflare_config_file, to_json=False)


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


def is_dir_empty(path: str):
    targe_dir = os.listdir(path)
    return len(targe_dir) == 0


def load_job_info_config(hidden_nvflare_config_file) -> ConfigTree:
    return CF.parse_file(hidden_nvflare_config_file)


def create_job_info_config(cmd_args, nvflare_config: ConfigTree) -> ConfigTree:
    """
    Args:
        cmd_args: Command-line arguments containing the startup directory path.
        nvflare_config (ConfigTree): The existing nvflare configuration.

    Returns:
        ConfigTree: The merged configuration tree.
    """
    startup_kit_dir = get_startup_kit_dir()
    if not startup_kit_dir or not os.path.isdir(startup_kit_dir):
        raise ValueError(f"startup_kit_dir '{startup_kit_dir}' must be a valid and non-empty path. "
                         f"use 'nvflare poc' command to 'prepare' if you are using POC mode. Or use"
                         f" 'nvflare config' to setup startup_kit_dir location if you are in production")

    conf_str = f"""
        startup_kit {{
            path = {startup_kit_dir}
        }}
    """
    conf: ConfigTree = CF.parse_string(conf_str)

    return conf.with_fallback(nvflare_config)


def get_hidden_nvflare_config_path() -> str:
    """
    Get the path for the hidden nvflare configuration file.

    Returns:
        str: The path to the hidden nvflare configuration file.
    """
    home_dir = get_home_dir()
    hidden_nvflare_dir = pathlib.Path(home_dir) / ".nvflare"

    try:
        hidden_nvflare_dir.mkdir(exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Error creating the hidden nvflare directory: {e}")

    hidden_nvflare_config_file = hidden_nvflare_dir / "config.conf"
    return str(hidden_nvflare_config_file)


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


def save_config(dst_config, dst_path, to_json=True):
    config_str = HOCONConverter.to_json(dst_config) if to_json else HOCONConverter.to_hocon(dst_config)

    with open(dst_path, "w") as outfile:
        outfile.write(f"{config_str}\n")


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

    file_dir = os.path.dirname(__file__)
    server_config = CF.parse_file(os.path.join(file_dir, "config/server_config.conf"))
    client_config = CF.parse_file(os.path.join(file_dir, "config/client_config.conf"))

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

        workflows.append(target_wf_conf)
        predefined_wf_components = workflows_conf.get(f"{wf_name}.components")
        predefined_wf_task_data_filters = workflows_conf.get(f"{wf_name}.task_data_filters", None)
        predefined_wf_task_result_filters = workflows_conf.get(f"{wf_name}.task_result_filters", None)

        if predefined_wf_task_data_filters:
            for name, data_filter in predefined_wf_task_data_filters.items():
                wf_task_task_data_filters.append(data_filter)

        if predefined_wf_task_result_filters:
            for name, result_filter in predefined_wf_task_result_filters.items():
                wf_task_result_filters.append(result_filter)

        for name, comp in predefined_wf_components.items():
            if name == "persistor":
                if cmd_args.enable_persistor:
                    wf_components.append(comp)
                    target_wf_conf.put("args.persistor_id", comp.get_string("id"))
            elif name == "model_selector":
                if cmd_args.enable_persistor:
                    wf_components.append(comp)
            else:
                wf_components.append(comp)

        predefined_executors = workflows_conf.get_config(f"{wf_name}.executors")
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
            executors.append(target_exec.get("executor"))
            if target_exec.get("task_data_filters", None):
                for name, data_filter in target_exec.get("task_data_filters").items():
                    exec_task_data_filters.append(data_filter)
            if target_exec.get("task_result_filters", None):
                for name, result_filter in target_exec.get("task_result_filters").items():
                    exec_task_result_filters.append(result_filter)
            for name, comp in target_exec.get("components").items():
                exec_components.append(comp)

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

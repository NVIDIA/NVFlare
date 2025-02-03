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

"""Provides a command line interface for federated server."""

import argparse
import logging
import os
import sys
import threading

from nvflare.apis.fl_constant import ConfigVarName, JobConstants, SiteType, SystemConfigs
from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.f3.mpm import MainProcessMonitor as mpm
from nvflare.fuel.sec.authn import set_add_auth_headers_filters
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.log_utils import configure_logging, get_script_logger
from nvflare.private.defs import AUTH_CLIENT_NAME_FOR_SJ, AppFolderConstants
from nvflare.private.fed.app.fl_conf import FLServerStarterConfiger
from nvflare.private.fed.app.utils import monitor_parent_process
from nvflare.private.fed.server.server_app_runner import ServerAppRunner
from nvflare.private.fed.server.server_state import HotState
from nvflare.private.fed.utils.fed_utils import (
    create_stats_pool_files_for_job,
    fobs_initialize,
    register_ext_decomposers,
    security_close,
    security_init_for_job,
    set_stats_pool_config_for_job,
)
from nvflare.security.logging import secure_format_exception, secure_log_traceback


def main(args):
    kv_list = parse_vars(args.set)

    config_folder = kv_list.get("config_folder", "")
    if config_folder == "":
        args.server_config = JobConstants.SERVER_JOB_CONFIG
    else:
        args.server_config = os.path.join(config_folder, JobConstants.SERVER_JOB_CONFIG)

    # TODO:: remove env and train config since they are not core
    args.env = os.path.join("config", AppFolderConstants.CONFIG_ENV)
    args.config_folder = config_folder
    args.log_config = None
    args.snapshot = kv_list.get("restore_snapshot")

    # get parent process id
    parent_pid = os.getppid()
    stop_event = threading.Event()
    workspace = Workspace(root_dir=args.workspace, site_name=SiteType.SERVER)
    set_stats_pool_config_for_job(workspace, args.job_id)
    secure_train = kv_list.get("secure_train", False)

    try:
        os.chdir(args.workspace)
        fobs_initialize(workspace=workspace, job_id=args.job_id)

        # initialize security processing and ensure that content in the startup has not been tampered with.
        security_init_for_job(secure_train, workspace, SiteType.SERVER)

        conf = FLServerStarterConfiger(
            workspace=workspace,
            args=args,
            kv_list=args.set,
        )
        configure_logging(workspace, workspace.get_run_dir(args.job_id))
        logger = get_script_logger()
        logger.info("Runner_process started.")

        log_level = os.environ.get("FL_LOG_LEVEL", "")
        numeric_level = getattr(logging, log_level.upper(), None)
        if isinstance(numeric_level, int):
            logging.getLogger().setLevel(numeric_level)
            logger.debug("loglevel debug enabled")
            logger.info("loglevel info enabled")
            logger.warning("loglevel warn enabled")
            logger.error("loglevel error enabled")
            logger.critical("loglevel critical enabled")

        conf.configure()
        event_handlers = conf.handlers
        deployer = conf.deployer

        decomposer_module = ConfigService.get_str_var(
            name=ConfigVarName.DECOMPOSER_MODULE, conf=SystemConfigs.RESOURCES_CONF
        )
        register_ext_decomposers(decomposer_module)

        try:
            # create the FL server
            server_config, server = deployer.create_fl_server(args, secure_train=secure_train)
            server.ha_mode = eval(args.ha_mode)

            server.cell = server.create_job_cell(
                args.job_id, args.root_url, args.parent_url, secure_train, server_config
            )

            # set filter to add additional auth headers
            set_add_auth_headers_filters(
                cell=server.cell,
                client_name=AUTH_CLIENT_NAME_FOR_SJ,
                auth_token=args.job_id,
                token_signature=args.token_signature,
                ssid=args.ssid,
            )

            server.server_state = HotState(host=args.host, port=args.port, ssid=args.ssid)

            snapshot = None
            if args.snapshot:
                snapshot = server.snapshot_persistor.retrieve_run(args.job_id)

            server_app_runner = ServerAppRunner(server)
            # start parent process checking thread
            thread = threading.Thread(target=monitor_parent_process, args=(server_app_runner, parent_pid, stop_event))
            thread.start()

            server_app_runner.start_server_app(
                workspace, args, args.app_root, args.job_id, snapshot, logger, args.set, event_handlers=event_handlers
            )
        finally:
            if deployer:
                deployer.close()
            stop_event.set()
            security_close()
            err = create_stats_pool_files_for_job(workspace, args.job_id)
            if err:
                logger.warning(err)

    except ConfigError as e:
        logger = get_script_logger()
        logger.exception(f"ConfigError: {secure_format_exception(e)}")
        secure_log_traceback(logger)
        raise e


def parse_arguments():
    """FL Server program starting point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-m", type=str, help="WORKSPACE folder", required=True)
    parser.add_argument("--fed_server", "-s", type=str, help="server config json file", required=True)
    parser.add_argument("--app_root", "-r", type=str, help="App Root", required=True)
    parser.add_argument("--job_id", "-n", type=str, help="job id", required=True)
    parser.add_argument("--token_signature", "-ts", type=str, help="auth token signature", required=True)
    parser.add_argument("--root_url", "-u", type=str, help="root_url", required=True)
    parser.add_argument("--host", "-host", type=str, help="server host", required=True)
    parser.add_argument("--port", "-port", type=str, help="service port", required=True)
    parser.add_argument("--ssid", "-id", type=str, help="SSID", required=True)
    parser.add_argument("--parent_url", "-p", type=str, help="parent_url", required=True)
    parser.add_argument("--ha_mode", "-ha_mode", type=str, help="HA mode", required=True)
    parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    This is the program when starting the child process for running the NVIDIA FLARE server runner.
    """
    # main()
    args = parse_arguments()
    run_dir = os.path.join(args.workspace, args.job_id)
    rc = mpm.run(main_func=main, run_dir=run_dir, args=args)
    sys.exit(rc)

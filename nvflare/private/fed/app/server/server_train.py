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

"""Federated server launching script."""

import argparse
import logging
import os
import sys
import time

from nvflare.apis.fl_constant import FLContextKey, JobConstants, SiteType, WorkspaceConstants
from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.f3.mpm import MainProcessMonitor as mpm
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.fuel.utils.log_utils import configure_logging
from nvflare.private.defs import AppFolderConstants
from nvflare.private.fed.app.fl_conf import FLServerStarterConfiger, create_privacy_manager
from nvflare.private.fed.app.utils import create_admin_server, version_check
from nvflare.private.fed.server.server_status import ServerStatus
from nvflare.private.fed.utils.fed_utils import fobs_initialize, security_init
from nvflare.private.privacy_manager import PrivacyService
from nvflare.security.logging import secure_format_exception


def main(args):
    kv_list = parse_vars(args.set)

    config_folder = kv_list.get("config_folder", "")
    if config_folder == "":
        args.server_config = JobConstants.SERVER_JOB_CONFIG
    else:
        args.server_config = os.path.join(config_folder, JobConstants.SERVER_JOB_CONFIG)

    # TODO:: remove env and train config since they are not core
    args.env = os.path.join("config", AppFolderConstants.CONFIG_ENV)
    args.train_config = os.path.join("config", AppFolderConstants.CONFIG_TRAIN)
    args.config_folder = config_folder
    logger = logging.getLogger()
    args.log_config = None
    args.job_id = None

    workspace = Workspace(root_dir=args.workspace, site_name=SiteType.SERVER)
    for name in [WorkspaceConstants.RESTART_FILE, WorkspaceConstants.SHUTDOWN_FILE]:
        try:
            f = workspace.get_file_path_in_root(name)
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            print(f"Could not remove file '{name}'.  Please check your system before starting FL.")
            sys.exit(-1)

    try:
        os.chdir(args.workspace)

        fobs_initialize(workspace)

        conf = FLServerStarterConfiger(
            workspace=workspace,
            args=args,
            kv_list=args.set,
        )

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

        configure_logging(workspace)

        deployer = conf.deployer
        secure_train = conf.cmd_vars.get("secure_train", False)

        security_init(
            secure_train=secure_train,
            site_org=conf.site_org,
            workspace=workspace,
            app_validator=conf.app_validator,
            site_type=SiteType.SERVER,
        )

        # initialize Privacy Service
        privacy_manager = create_privacy_manager(workspace, names_only=True, is_server=True)
        PrivacyService.initialize(privacy_manager)

        try:
            # Deploy the FL server
            services = deployer.deploy(args)

            first_server = sorted(conf.config_data["servers"])[0]
            admin_server = create_admin_server(
                services,
                server_conf=first_server,
                args=args,
            )
            admin_server.start()
            services.set_admin_server(admin_server)

            with services.engine.new_context() as fl_ctx:
                fl_ctx.set_prop(
                    key=FLContextKey.SERVER_CONFIG,
                    value=deployer.server_config,
                    private=True,
                    sticky=True,
                )

                fl_ctx.set_prop(
                    key=FLContextKey.SECURE_MODE,
                    value=deployer.secure_train,
                    private=True,
                    sticky=True,
                )

        finally:
            deployer.close()

        logger.info("Server started")

        # From Python 3.9 and above, the ThreadPoolExecutor does not allow submit() to create a new thread while the
        # main thread has exited. Use the ServerStatus.SHUTDOWN to keep the main thread waiting for the gRPC
        # server to be shutdown.
        while services.status != ServerStatus.SHUTDOWN:
            time.sleep(1.0)

        if admin_server:
            admin_server.stop()
        services.engine.close()

    except ConfigError as e:
        logger.exception(f"ConfigError: {secure_format_exception(e)}")
        raise e


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-m", type=str, help="WORKSPACE folder", required=True)
    parser.add_argument(
        "--fed_server", "-s", type=str, help="an aggregation server specification json file", required=True
    )
    parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    This is the main program when starting the NVIDIA FLARE server process.
    """

    version_check()
    args = parse_arguments()
    rc = mpm.run(main_func=main, run_dir=args.workspace, args=args)
    sys.exit(rc)

# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.fl_constant import JobConstants, SiteType, WorkspaceConstants
from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.hci.security import hash_password
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.private.defs import AppFolderConstants, SSLConstants
from nvflare.private.fed.app.fl_conf import FLServerStarterConfiger, create_privacy_manager
from nvflare.private.fed.server.admin import FedAdminServer
from nvflare.private.fed.server.fed_server import FederatedServer
from nvflare.private.fed.utils.fed_utils import add_logfile_handler, fobs_initialize, security_init
from nvflare.private.privacy_manager import PrivacyService
from nvflare.security.logging import secure_format_exception


def main():
    if sys.version_info >= (3, 9):
        raise RuntimeError("Python versions 3.9 and above are not yet supported. Please use Python 3.8 or 3.7.")
    if sys.version_info < (3, 7):
        raise RuntimeError("Python versions 3.6 and below are not supported. Please use Python 3.8 or 3.7.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-m", type=str, help="WORKSPACE folder", required=True)
    parser.add_argument(
        "--fed_server", "-s", type=str, help="an aggregation server specification json file", required=True
    )
    parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")

    args = parser.parse_args()
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

    workspace = Workspace(root_dir=args.workspace, site_name="server")
    for name in [WorkspaceConstants.RESTART_FILE, WorkspaceConstants.SHUTDOWN_FILE]:
        try:
            f = workspace.get_file_path_in_root(name)
            if os.path.exists(f):
                os.remove(f)
        except BaseException:
            print(f"Could not remove file '{name}'.  Please check your system before starting FL.")
            sys.exit(-1)

    try:
        os.chdir(args.workspace)

        fobs_initialize()

        conf = FLServerStarterConfiger(
            workspace=workspace,
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

        log_file = workspace.get_log_file_path()
        add_logfile_handler(log_file)

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
        privacy_manager = create_privacy_manager(workspace, names_only=True)
        PrivacyService.initialize(privacy_manager)

        try:
            # Deploy the FL server
            services = deployer.deploy(args)

            first_server = sorted(conf.config_data["servers"])[0]
            # allow command to overwrite the admin_host
            if conf.cmd_vars.get("host", None):
                first_server["admin_host"] = conf.cmd_vars["host"]
            admin_server = create_admin_server(
                services,
                server_conf=first_server,
                args=args,
                secure_train=secure_train,
            )
            admin_server.start()
            services.set_admin_server(admin_server)
        finally:
            deployer.close()

        logger.info("Server started")

    except ConfigError as e:
        logger.exception(f"ConfigError: {secure_format_exception(e)}")
        raise e


def create_admin_server(fl_server: FederatedServer, server_conf=None, args=None, secure_train=False):
    """To create the admin server.

    Args:
        fl_server: fl_server
        server_conf: server config
        args: command args
        secure_train: True/False

    Returns:
        A FedAdminServer.
    """
    users = {}
    # Create a default user admin:admin for the POC insecure use case.
    if not secure_train:
        users = {"admin": hash_password("admin")}

    root_cert = server_conf[SSLConstants.ROOT_CERT] if secure_train else None
    server_cert = server_conf[SSLConstants.CERT] if secure_train else None
    server_key = server_conf[SSLConstants.PRIVATE_KEY] if secure_train else None
    admin_server = FedAdminServer(
        fed_admin_interface=fl_server.engine,
        users=users,
        cmd_modules=fl_server.cmd_modules,
        file_upload_dir=os.path.join(args.workspace, server_conf.get("admin_storage", "tmp")),
        file_download_dir=os.path.join(args.workspace, server_conf.get("admin_storage", "tmp")),
        host=server_conf.get("admin_host", "localhost"),
        port=server_conf.get("admin_port", 5005),
        ca_cert_file_name=root_cert,
        server_cert_file_name=server_cert,
        server_key_file_name=server_key,
        accepted_client_cns=None,
        download_job_url=server_conf.get("download_job_url", "http://"),
    )
    return admin_server


if __name__ == "__main__":
    """
    This is the main program when starting the NVIDIA FLARE server process.
    """

    main()

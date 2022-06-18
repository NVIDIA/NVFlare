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

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.hci.security import hash_password
from nvflare.fuel.hci.server.authz import AuthorizationService
from nvflare.fuel.sec.audit import AuditService
from nvflare.fuel.sec.security_content_service import SecurityContentService
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.private.defs import AppFolderConstants, SSLConstants
from nvflare.private.fed.app.fl_conf import FLServerStarterConfiger
from nvflare.private.fed.server.admin import FedAdminServer
from nvflare.private.fed.server.fed_server import FederatedServer
from nvflare.private.fed.utils.fed_utils import add_logfile_handler, secure_content_check
from nvflare.security.security import EmptyAuthorizer, FLAuthorizer


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
        args.server_config = AppFolderConstants.CONFIG_FED_SERVER
    else:
        args.server_config = os.path.join(config_folder, AppFolderConstants.CONFIG_FED_SERVER)

    # TODO:: remove env and train config since they are not core
    args.env = os.path.join("config", AppFolderConstants.CONFIG_ENV)
    args.train_config = os.path.join("config", AppFolderConstants.CONFIG_TRAIN)
    args.config_folder = config_folder
    logger = logging.getLogger()
    args.log_config = None

    for name in [WorkspaceConstants.RESTART_FILE, WorkspaceConstants.SHUTDOWN_FILE]:
        try:
            f = os.path.join(args.workspace, name)
            if os.path.exists(f):
                os.remove(f)
        except BaseException:
            print("Could not remove the {} file.  Please check your system before starting FL.".format(name))
            sys.exit(-1)

    try:
        os.chdir(args.workspace)

        startup = os.path.join(args.workspace, "startup")
        conf = FLServerStarterConfiger(
            app_root=startup,
            server_config_file_name=args.fed_server,
            log_config_file_name=WorkspaceConstants.LOGGING_CONFIG,
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

        log_file = os.path.join(args.workspace, "log.txt")
        add_logfile_handler(log_file)

        deployer = conf.deployer
        secure_train = conf.cmd_vars.get("secure_train", False)

        security_check(secure_train=secure_train, content_folder=startup, fed_server_config=args.fed_server)

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
                app_validator=deployer.app_validator,
            )
            admin_server.start()

            services.platform = "PT"

            services.set_admin_server(admin_server)
        finally:
            deployer.close()

        logger.info("Server started")

    except ConfigError as ex:
        print("ConfigError:", str(ex))
    finally:
        pass


def security_check(secure_train: bool, content_folder: str, fed_server_config: str):
    """To check the security content if running in security mode.

    Args:
        secure_train (bool): if run in secure mode or not.
        content_folder (str): the folder to check.
        fed_server_config (str): fed_server.json
    """
    # initialize the SecurityContentService.
    # must do this before initializing other services since it may be needed by them!
    SecurityContentService.initialize(content_folder=content_folder)

    if secure_train:
        insecure_list = secure_content_check(fed_server_config, site_type="server")
        if len(insecure_list):
            print("The following files are not secure content.")
            for item in insecure_list:
                print(item)
            sys.exit(1)

    # initialize the AuditService, which is used by command processing.
    # The Audit Service can be used in other places as well.
    AuditService.initialize(audit_file_name=WorkspaceConstants.AUDIT_LOG)

    # Initialize the AuthorizationService. It is used by command authorization
    # We use FLAuthorizer for policy processing.
    # AuthorizationService depends on SecurityContentService to read authorization policy file.
    if secure_train:
        _, err = AuthorizationService.initialize(FLAuthorizer())
    else:
        _, err = AuthorizationService.initialize(EmptyAuthorizer())

    if err:
        print("AuthorizationService error: {}".format(err))
        sys.exit(1)


def create_admin_server(
    fl_server: FederatedServer, server_conf=None, args=None, secure_train=False, app_validator=None
):
    """To create the admin server.

    Args:
        fl_server: fl_server
        server_conf: server config
        args: command args
        secure_train: True/False
        app_validator: application validator

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
        allowed_shell_cmds=None,
        host=server_conf.get("admin_host", "localhost"),
        port=server_conf.get("admin_port", 5005),
        ca_cert_file_name=root_cert,
        server_cert_file_name=server_cert,
        server_key_file_name=server_key,
        accepted_client_cns=None,
        app_validator=app_validator,
        download_job_url=server_conf.get("download_job_url", "http://"),
    )
    return admin_server


if __name__ == "__main__":
    """
    This is the main program when starting the NVIDIA FLARE server process.
    """

    main()

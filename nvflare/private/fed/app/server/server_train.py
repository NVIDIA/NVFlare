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

"""Provides a command line interface for federated server."""

import argparse
import logging
import os
import sys

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.hci.server.authz import AuthorizationService
from nvflare.fuel.sec.audit import AuditService
from nvflare.fuel.sec.security_content_service import LoadResult, SecurityContentService
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.private.fed.app.fl_conf import FLServerStarterConfiger
from nvflare.private.fed.server.admin import FedAdminServer
from nvflare.security.security import EmptyAuthorizer, FLAuthorizer


def main():
    """FL Server program starting point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-m", type=str, help="WORKSPACE folder", required=True)

    parser.add_argument(
        "--fed_server", "-s", type=str, help="an aggregation server specification json file", required=True
    )

    parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")

    args = parser.parse_args()
    kv_list = parse_vars(args.set)

    args.train_config = "config/config_train.json"
    config_folder = kv_list.get("config_folder", "")
    if config_folder == "":
        args.server_config = "config_fed_server.json"
    else:
        args.server_config = config_folder + "/config_fed_server.json"
    args.env = "config/environment.json"
    args.config_folder = config_folder
    logger = logging.getLogger()
    args.log_config = None

    try:
        remove_restart_file(args)
    except BaseException:
        print("Could not remove the restart.fl / shutdown.fl file.  Please check your system before starting FL.")
        sys.exit(-1)

    try:
        os.chdir(args.workspace)

        # trainer = WorkFlowFactory().create_server_trainer(train_configs, envs)
        startup = os.path.join(args.workspace, "startup")
        conf = FLServerStarterConfiger(
            app_root=startup,
            # wf_config_file_name="config_train.json",
            server_config_file_name=args.fed_server,
            # env_config_file_name="environment.json",
            log_config_file_name="log.config",
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

        deployer = conf.deployer
        secure_train = conf.cmd_vars.get("secure_train", False)

        security_check(secure_train, args)

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


def security_check(secure_train, args):
    """To check the security content if running in security mode.

    Args:
        secure_train: True/False
        args: command args

    """
    # initialize the SecurityContentService.
    # must do this before initializing other services since it may be needed by them!
    startup = os.path.join(args.workspace, "startup")
    SecurityContentService.initialize(content_folder=startup)

    if secure_train:
        insecure_list = secure_content_check(args)
        if len(insecure_list):
            print("The following files are not secure content.")
            for item in insecure_list:
                print(item)
            sys.exit(1)

    # initialize the AuditService, which is used by command processing.
    # The Audit Service can be used in other places as well.
    AuditService.initialize(audit_file_name="audit.log")
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


def secure_content_check(args):
    """To check the security contents.

    Args:
        args: command args

    Returns: the insecure content list

    """
    insecure_list = []
    data, sig = SecurityContentService.load_json(args.fed_server)
    if sig != LoadResult.OK:
        insecure_list.append(args.fed_server)

    for server in data["servers"]:
        content, sig = SecurityContentService.load_content(server.get("ssl_cert"))
        if sig != LoadResult.OK:
            insecure_list.append(server.get("ssl_cert"))
        content, sig = SecurityContentService.load_content(server.get("ssl_private_key"))
        if sig != LoadResult.OK:
            insecure_list.append(server.get("ssl_private_key"))
        content, sig = SecurityContentService.load_content(server.get("ssl_root_cert"))
        if sig != LoadResult.OK:
            insecure_list.append(server.get("ssl_root_cert"))

    if "authorization.json" in SecurityContentService.security_content_manager.signature:
        data, sig = SecurityContentService.load_json("authorization.json")
        if sig != LoadResult.OK:
            insecure_list.append("authorization.json")

    return insecure_list


def remove_restart_file(args):
    """To remove the restart.fl file.

    Args:
        args: command args

    """
    restart_file = os.path.join(args.workspace, "restart.fl")
    if os.path.exists(restart_file):
        os.remove(restart_file)
    restart_file = os.path.join(args.workspace, "shutdown.fl")
    if os.path.exists(restart_file):
        os.remove(restart_file)


def create_admin_server(fl_server, server_conf=None, args=None, secure_train=False, app_validator=None):
    """To create the admin server.

    Args:
        fl_server: fl_server
        server_conf: server config
        args: command args
        secure_train: True/False
        app_validator: application validator

    Returns: admin server

    """
    # sai = ServerEngine(fl_server, args)
    users = {}
    # Create a default user admin:admin for the POC insecure use case.
    if not secure_train:
        users = {
            "admin": "e7b71aa322cecc502e9454271b98feaec594da944c369facc90ac85016dc6c74c3fd99657ebd9d083a7804c3a17ddd8c655df8bcbf172be9d0207c8c9430c19be3cd846949505d283e066434175956bf45cd1d6781e63e5be4f3e23533d4d002"
        }
    # cmd_modules = [ValidationCommandModule()]

    root_cert = server_conf["ssl_root_cert"] if secure_train else None
    server_cert = server_conf["ssl_cert"] if secure_train else None
    server_key = server_conf["ssl_private_key"] if secure_train else None
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
    )
    return admin_server


if __name__ == "__main__":
    """
    This is the main program when starting the NVIDIA FLARE server process.
    """

    main()

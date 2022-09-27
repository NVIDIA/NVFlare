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

import json
import logging
import logging.config
import os
import sys
from logging.handlers import RotatingFileHandler
from multiprocessing.connection import Listener
from typing import List

from nvflare.apis.app_validation import AppValidator
from nvflare.apis.fl_constant import FLContextKey, SiteType, WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.apis.workspace import Workspace
from nvflare.app_common.decomposers import common_decomposers
from nvflare.fuel.sec.audit import AuditService
from nvflare.fuel.sec.authz import AuthorizationService
from nvflare.fuel.sec.security_content_service import LoadResult, SecurityContentService
from nvflare.fuel.utils import fobs
from nvflare.private.defs import SSLConstants
from nvflare.private.fed.protos.federated_pb2 import ModelData
from nvflare.private.fed.utils.numproto import bytes_to_proto
from nvflare.private.privacy_manager import PrivacyManager, PrivacyService
from nvflare.security.logging import secure_format_exception, secure_log_traceback
from nvflare.security.security import EmptyAuthorizer, FLAuthorizer

from .app_authz import AppAuthzService


def shareable_to_modeldata(shareable, fl_ctx):
    # make_init_proto message
    model_data = ModelData()  # create an empty message

    model_data.params["data"].CopyFrom(make_shareable_data(shareable))

    context_data = make_context_data(fl_ctx)
    model_data.params["fl_context"].CopyFrom(context_data)
    return model_data


def make_shareable_data(shareable):
    return bytes_to_proto(shareable.to_bytes())


def make_context_data(fl_ctx):
    shared_fl_ctx = FLContext()
    shared_fl_ctx.set_public_props(fl_ctx.get_all_public_props())
    props = fobs.dumps(shared_fl_ctx)
    context_data = bytes_to_proto(props)
    return context_data


def add_logfile_handler(log_file):
    root_logger = logging.getLogger()
    main_handler = root_logger.handlers[0]
    file_handler = RotatingFileHandler(log_file, maxBytes=20 * 1024 * 1024, backupCount=10)
    file_handler.setLevel(main_handler.level)
    file_handler.setFormatter(main_handler.formatter)
    root_logger.addHandler(file_handler)


def listen_command(listen_port, engine, execute_func, logger):
    conn = None
    listener = None
    try:
        address = ("localhost", listen_port)
        listener = Listener(address, authkey="client process secret password".encode())
        conn = listener.accept()

        execute_func(conn, engine)

    except Exception as e:
        logger.exception(
            f"Could not create the listener for this process on port: {listen_port}: {secure_format_exception(e)}."
        )
        secure_log_traceback(logger)
    finally:
        if conn:
            conn.close()
        if listener:
            listener.close()


def _check_secure_content(site_type: str) -> List[str]:
    """To check the security contents.

    Args:
        site_type (str): "server" or "client"

    Returns:
        A list of insecure content.
    """
    if site_type == SiteType.SERVER:
        config_file_name = WorkspaceConstants.SERVER_STARTUP_CONFIG
    else:
        config_file_name = WorkspaceConstants.CLIENT_STARTUP_CONFIG

    insecure_list = []
    data, sig = SecurityContentService.load_json(config_file_name)
    if sig != LoadResult.OK:
        insecure_list.append(config_file_name)

    sites_to_check = data["servers"] if site_type == SiteType.SERVER else [data["client"]]

    for site in sites_to_check:
        for filename in [SSLConstants.CERT, SSLConstants.PRIVATE_KEY, SSLConstants.ROOT_CERT]:
            content, sig = SecurityContentService.load_content(site.get(filename))
            if sig != LoadResult.OK:
                insecure_list.append(site.get(filename))

    if WorkspaceConstants.AUTHORIZATION_CONFIG in SecurityContentService.security_content_manager.signature:
        data, sig = SecurityContentService.load_json(WorkspaceConstants.AUTHORIZATION_CONFIG)
        if sig != LoadResult.OK:
            insecure_list.append(WorkspaceConstants.AUTHORIZATION_CONFIG)

    return insecure_list


def security_init(secure_train: bool, site_org: str, workspace: Workspace, app_validator: AppValidator, site_type: str):
    """To check the security content if running in security mode.

    Args:
       secure_train (bool): if run in secure mode or not.
       site_org: organization of the site
       workspace: the workspace object.
       app_validator: app validator for application validation
       site_type (str): server or client. fed_client.json or fed_server.json
    """
    # initialize the SecurityContentService.
    # must do this before initializing other services since it may be needed by them!
    startup_dir = workspace.get_startup_kit_dir()
    SecurityContentService.initialize(content_folder=startup_dir)

    if secure_train:
        insecure_list = _check_secure_content(site_type=site_type)
        if len(insecure_list):
            print("The following files are not secure content.")
            for item in insecure_list:
                print(item)
            sys.exit(1)

    # initialize the AuditService, which is used by command processing.
    # The Audit Service can be used in other places as well.
    audit_file_name = workspace.get_audit_file_path()
    AuditService.initialize(audit_file_name)

    if app_validator:
        AppAuthzService.initialize(app_validator)

    # Initialize the AuthorizationService. It is used by command authorization
    # We use FLAuthorizer for policy processing.
    # AuthorizationService depends on SecurityContentService to read authorization policy file.
    authorizer = None
    if secure_train:
        policy_file_path = workspace.get_authorization_file_path()

        if policy_file_path and os.path.exists(policy_file_path):
            policy_config = json.load(open(policy_file_path, "rt"))
            authorizer = FLAuthorizer(site_org, policy_config)

    if not authorizer:
        authorizer = EmptyAuthorizer()

    _, err = AuthorizationService.initialize(authorizer)

    if err:
        print("AuthorizationService error: {}".format(err))
        sys.exit(1)


def security_close():
    AuditService.close()


def get_job_meta_from_workspace(workspace: Workspace, job_id: str) -> dict:
    job_meta_file_path = workspace.get_job_meta_path(job_id)
    with open(job_meta_file_path) as file:
        return json.load(file)


def create_job_processing_context_properties(workspace: Workspace, job_id: str) -> dict:
    job_meta = get_job_meta_from_workspace(workspace, job_id)
    assert isinstance(job_meta, dict), f"job_meta must be dict but got {type(job_meta)}"
    scope_name = job_meta.get(JobMetaKey.SCOPE, "")
    scope_object = PrivacyService.get_scope(scope_name)
    scope_props = None
    if scope_object:
        scope_props = scope_object.props
        effective_scope_name = scope_object.name
    else:
        effective_scope_name = ""

    return {
        FLContextKey.JOB_META: job_meta,
        FLContextKey.JOB_SCOPE_NAME: scope_name,
        FLContextKey.EFFECTIVE_JOB_SCOPE_NAME: effective_scope_name,
        FLContextKey.SCOPE_PROPERTIES: scope_props,
        FLContextKey.SCOPE_OBJECT: scope_object,
    }


def configure_logging(workspace: Workspace):
    log_config_file_path = workspace.get_log_config_file_path()
    assert os.path.isfile(log_config_file_path), f"missing log config file {log_config_file_path}"
    logging.config.fileConfig(fname=log_config_file_path, disable_existing_loggers=False)


def get_scope_info():
    try:
        privacy_manager = PrivacyService.get_manager()
        scope_names = []
        default_scope_name = ""
        if privacy_manager:
            assert isinstance(privacy_manager, PrivacyManager)
            if privacy_manager.name_to_scopes:
                scope_names = sorted(privacy_manager.name_to_scopes.keys(), reverse=False)
            if privacy_manager.default_scope:
                default_scope_name = privacy_manager.default_scope.name
        return scope_names, default_scope_name
    except:
        return [], "processing_error"


def fobs_initialize():
    flare_decomposers.register()
    common_decomposers.register()

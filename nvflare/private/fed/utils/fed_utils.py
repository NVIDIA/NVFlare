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

import logging
import os
import pickle
import shutil
from logging.handlers import RotatingFileHandler
from multiprocessing.connection import Listener
from typing import List

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.hci.zip_utils import unzip_all_from_bytes
from nvflare.fuel.sec.security_content_service import LoadResult, SecurityContentService
from nvflare.private.defs import SSLConstants
from nvflare.private.fed.protos.federated_pb2 import ModelData
from nvflare.private.fed.utils.numproto import bytes_to_proto


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
    props = pickle.dumps(shared_fl_ctx)
    context_data = bytes_to_proto(props)
    return context_data


def deploy_app(app_name, site_name, workspace, app_data):
    try:
        dest = os.path.join(workspace, WorkspaceConstants.APP_PREFIX + site_name)
        # Remove the previous deployed app.
        if os.path.exists(dest):
            shutil.rmtree(dest)
        if not os.path.exists(dest):
            os.makedirs(dest)
        unzip_all_from_bytes(app_data, dest)
        app_file = os.path.join(workspace, "fl_app.txt")
        if os.path.exists(app_file):
            os.remove(app_file)
        with open(app_file, "wt") as f:
            f.write(f"{app_name}")
        return True
    except:
        return False


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
        logger.exception(f"Could not create the listener for this process on port: {listen_port}: {e}.", exc_info=True)
    finally:
        if conn:
            conn.close()
        if listener:
            listener.close()


def secure_content_check(config: str, site_type: str) -> List[str]:
    """To check the security contents.

    Args:
        config (str): The fed_XXX config
        site_type (str): "server" or "client"

    Returns:
        A list of insecure content.
    """
    insecure_list = []
    data, sig = SecurityContentService.load_json(config)
    if sig != LoadResult.OK:
        insecure_list.append(config)

    sites_to_check = data["servers"] if site_type == "server" else [data["client"]]

    for site in sites_to_check:
        for filename in [SSLConstants.CERT, SSLConstants.PRIVATE_KEY, SSLConstants.ROOT_CERT]:
            content, sig = SecurityContentService.load_content(site.get(filename))
            if sig != LoadResult.OK:
                insecure_list.append(site.get(filename))

    if site_type == "server":
        if "authorization.json" in SecurityContentService.security_content_manager.signature:
            data, sig = SecurityContentService.load_json("authorization.json")
            if sig != LoadResult.OK:
                insecure_list.append("authorization.json")

    return insecure_list

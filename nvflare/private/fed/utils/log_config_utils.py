# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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
import logging.config
import os
from typing import Optional

from nvflare.apis.fl_constant import WorkspaceConstants


def initialize_log_config(workspace: str, resource_dir: str, log_file: Optional[str] = None):
    config_file_path = get_config_file_path(workspace, resource_dir)
    config_logging_by_file(config_file_path, log_file)


def config_logging_by_file(config_file_path: str , log_file: Optional[str] = None):
    config_schema = get_log_config_schema(config_file_path, log_file)
    logging.config.dictConfig(config_schema)
    logging.info(f"Log config is loaded from '{config_file_path}'")


def get_log_config_schema(config_file_path: str, log_file: Optional[str] = None) -> dict:
    import json

    config_schema = {}
    with open(config_file_path, encoding="utf-8") as config_file:
        log_config_str = config_file.read()
        config_schema = json.loads(log_config_str)

    if not config_schema:
        raise ValueError(f"invalid log_config_file: {config_file_path}")

    # update log name
    if log_file:
        handlers_config = config_schema["handlers"]
        for handler_name in handlers_config:
            one_handler_config = handlers_config[handler_name]
            if "filename" in one_handler_config:
                one_handler_config["filename"] = log_file

    return config_schema


def get_config_file_path(workspace: str, resource_dir: str):
    local_config_path = os.path.join(workspace, "local", WorkspaceConstants.LOGGING_CONFIG)
    if not os.path.isfile(local_config_path):
        config_file_path = os.path.join(resource_dir, WorkspaceConstants.LOGGING_CONFIG)
    else:
        config_file_path = local_config_path
    return config_file_path

#  Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import logging

from nvflare.fuel.utils.config_service import ConfigService


_comm_config_files = [
    "comm_config.json",
    "comm_config.json.default"
]


DEFAULT_MAX_MSG_SIZE = 1000 * 1024 * 1024    # 1000M

_KEY_MAX_MSG_SIZE = "max_message_size"


class CommConfigurator:

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        config = None
        for file_name in _comm_config_files:
            try:
                config = ConfigService.load_json(file_name)
                if config:
                    break
            except FileNotFoundError:
                self.logger.debug(f"config file {file_name} not found from config path")
                config = None
            except BaseException as ex:
                self.logger.error(f"failed to load config file {file_name}: {ex}")
                config = None
        self.config = config

    def get_config(self):
        return self.config

    def get_max_message_size(self):
        if self.config:
            return self.config.get("max_message_size", DEFAULT_MAX_MSG_SIZE)
        else:
            return DEFAULT_MAX_MSG_SIZE

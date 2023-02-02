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

from typing import Union
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.f3.cellnet.constants import ConnectorRequirementKey
from nvflare.fuel.utils.config_service import ConfigService

_KEY_RESOURCES = "resources"
_KEY_INT = "internal"
_KEY_EXT = "external_driver"
_KEY_SCHEME = "scheme"
_KEY_HOST = "host"
_KEY_PORTS = "ports"


class ConnectorInfo:

    def __init__(self, handle, connect_url: str, active: bool):
        self.handle = handle
        self.connect_url = connect_url
        self.active = active

    def get_connection_url(self):
        return self.connect_url


class ConnectorManager:

    """
    Manages creation of connectors
    """

    comm_config_files = [
        "comm_config.json",
        "comm_config.json.default"
    ]

    def __init__(
            self,
            communicator,
            secure: bool
    ):
        self.communicator = communicator
        self.secure = secure

        # set up default drivers
        self.int_scheme = "http"
        self.int_resources = {
            _KEY_HOST: "localhost",
            _KEY_PORTS: []  # select a port randomly
        }
        self.ext_scheme = "http"
        self.ext_resources = {}

        # load config if any
        config = None
        for file_name in self.comm_config_files:
            try:
                config = ConfigService.load_json(file_name)
            except FileNotFoundError:
                config = None

        if config:
            int_conf = self._validate_conn_config(config, _KEY_INT)
            if int_conf:
                self.int_scheme = int_conf.get(_KEY_SCHEME)
                self.int_resources = int_conf.get(_KEY_RESOURCES)

            ext_conf = self._validate_conn_config(config, _KEY_EXT)
            if ext_conf:
                self.ext_scheme = ext_conf.get(_KEY_SCHEME)
                self.ext_resources = ext_conf.get(_KEY_RESOURCES)

    def _validate_conn_config(self, config: dict, key: str) -> Union[None, dict]:
        conn_config = config.get(key)
        if conn_config:
            if not isinstance(conn_config, dict):
                raise ConfigError(f"'{key}' must be dict but got {type(conn_config)}")
            scheme = conn_config.get(_KEY_SCHEME)
            if not scheme:
                raise ConfigError(f"missing '{_KEY_SCHEME}' in {key} config")

            resources = conn_config.get(_KEY_RESOURCES)
            if resources:
                if not isinstance(resources, dict):
                    raise ConfigError(f"'{_KEY_RESOURCES}' in {key} must be dict but got {type(resources)}")
        return conn_config

    def _get_connector(self, url: str, active: bool, internal: bool, adhoc: bool) -> Union[None, ConnectorInfo]:
        if not adhoc:
            # backbone
            if not internal:
                # external
                scheme = self.ext_scheme
                resources = self.ext_resources
            else:
                # internal
                scheme = self.int_scheme
                resources = self.int_resources
        else:
            # ad-hoc - must be external
            if internal:
                raise RuntimeError("internal ad-hoc connector not supported")
            scheme = self.ext_scheme
            resources = self.ext_resources

        reqs = {
            ConnectorRequirementKey.URL: url,
            ConnectorRequirementKey.SECURE: self.secure
        }
        reqs.update(resources)
        active_url, passive_url = self.communicator.get_connector_url(scheme, reqs)
        if active:
            handle = self.communicator.add_connector(active_url, Mode.ACTIVE)
        else:
            handle = self.communicator.add_connector(passive_url, Mode.PASSIVE)

        return ConnectorInfo(handle, active_url, active)

    def get_external_listener(
            self,
            url: str,
            adhoc: bool
    ) -> Union[None, ConnectorInfo]:
        """
        Try to get an external listener.

        Args:
            url:
            adhoc:
        """
        return self._get_connector(
            url=url,
            active=False,
            internal=False,
            adhoc=adhoc
        )

    def get_external_connector(
            self,
            url: str,
            adhoc: bool
    ) -> Union[None, ConnectorInfo]:
        """
        Try to get an external listener.

        Args:
            url:
            adhoc:
        """
        return self._get_connector(
            url=url,
            active=True,
            internal=False,
            adhoc=adhoc
        )

    def get_internal_listener(self) -> Union[None, ConnectorInfo]:
        """
        Try to get an internal listener.
        """
        return self._get_connector(
            url="",
            active=False,
            internal=True,
            adhoc=False
        )

    def get_internal_connector(
            self,
            url: str
    ) -> Union[None, ConnectorInfo]:
        """
        Try to get an internal listener.

        Args:
            url:
        """
        return self._get_connector(
            url=url,
            active=True,
            internal=True,
            adhoc=False
        )

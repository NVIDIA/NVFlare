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
import os
import time
import traceback
from typing import Union
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.f3.communicator import Communicator, Mode, CommError
from nvflare.fuel.f3.comm_config import CommConfigurator
from .defs import ConnectorRequirementKey
from .fqcn import FqcnInfo

_KEY_RESOURCES = "resources"
_KEY_INT = "internal"
_KEY_EXT = "external"
_KEY_SCHEME = "scheme"
_KEY_HOST = "host"
_KEY_PORTS = "ports"
_KEY_ALLOW_ADHOC = "allow_adhoc"


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

    def __init__(
            self,
            communicator: Communicator,
            secure: bool,
            comm_configurator: CommConfigurator
    ):
        self._name = self.__class__.__name__
        self.logger = logging.getLogger(self._name)

        self.communicator = communicator
        self.secure = secure

        # set up default drivers
        self.int_scheme = "tcp"
        self.int_resources = {
            _KEY_HOST: "localhost",
            _KEY_PORTS: ["30000-40000"]  # select a port randomly
        }
        self.ext_allow_adhoc = False
        self.ext_scheme = "http"
        self.ext_resources = {}

        # load config if any
        comm_config = comm_configurator.get_config()
        if comm_config:
            int_conf = self._validate_conn_config(comm_config, _KEY_INT)
            if int_conf:
                self.int_scheme = int_conf.get(_KEY_SCHEME)
                self.int_resources = int_conf.get(_KEY_RESOURCES)

            ext_conf = self._validate_conn_config(comm_config, _KEY_EXT)
            if ext_conf:
                self.ext_scheme = ext_conf.get(_KEY_SCHEME)
                self.ext_resources = ext_conf.get(_KEY_RESOURCES)
                self.ext_allow_adhoc = ext_conf.get(_KEY_ALLOW_ADHOC, False)

        self.logger.debug(f"internal scheme={self.int_scheme}, resources={self.int_resources}")
        self.logger.debug(f"external scheme={self.ext_scheme}, resources={self.ext_resources}")
        self.comm_config = comm_config

    def should_connect_to_server(self, fqcn_info: FqcnInfo) -> bool:
        if fqcn_info.gen == 1:
            return True

        if self.comm_config:
            bb_config = self.comm_config.get("backbone")
            if bb_config:
                gens = bb_config.get("connect_generation")
                if gens:
                    if isinstance(gens, list):
                        return fqcn_info.gen in gens
                    else:
                        return fqcn_info.gen == gens
        # use default policy
        return fqcn_info.gen <= 2

    def is_adhoc_allowed(self, c1: FqcnInfo, c2: FqcnInfo) -> bool:
        """
        Is adhoc connection allowed between the two cells?
        Args:
            c1:
            c2:

        Returns:

        """
        if c1.root == c2.root:
            # same family
            return False

        if not self.ext_allow_adhoc:
            return False

        # we only allow gen2 (or above) cells to directly connect
        if c1.gen >= 2 and c2.gen >= 2:
            return True
        return False

    @staticmethod
    def _validate_conn_config(config: dict, key: str) -> Union[None, dict]:
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

    def _get_connector(
            self,
            url: str,
            active: bool,
            internal: bool,
            adhoc: bool,
            secure: bool
    ) -> Union[None, ConnectorInfo]:
        if active and not url:
            raise RuntimeError("url is required by not provided for active connector!")

        if not adhoc:
            # backbone
            if not internal:
                # external
                if not url:
                    raise RuntimeError("url is required but not provided for external backbone connector/listener!")
                scheme = self.ext_scheme
                resources = {}
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
            self.logger.debug(
                f"{os.getpid()}: creating ad-hoc external listener: "
                f"active={active} scheme={scheme}, resources={resources}")
            if not active and not self.ext_allow_adhoc:
                # ad-hoc listener is not allowed!
                return None

        reqs = {
            ConnectorRequirementKey.SECURE: secure
        }
        if url:
            reqs[ConnectorRequirementKey.URL] = url

        reqs.update(resources)

        try:
            if active:
                handle = self.communicator.add_connector(url, Mode.ACTIVE)
                connect_url = url
            elif url:
                handle = self.communicator.add_connector(url, Mode.PASSIVE)
                connect_url = url
            else:
                self.logger.info(f"{os.getpid()}: Try start_listener Listener resources: {reqs}")
                handle, connect_url = self.communicator.start_listener(scheme, reqs)
                self.logger.debug(f"{os.getpid()}: ############ dynamic listener at {connect_url}")
                # Kludge: to wait for listener ready and avoid race
                time.sleep(0.5)

            return ConnectorInfo(handle, connect_url, active)
        except CommError as ex:
            self.logger.error(f"Failed to get connector: {ex}")
            return None
        except:
            self.logger.error(f"unexpected exception:")
            self.logger.error(traceback.format_exc())
            return None

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
            adhoc=adhoc,
            secure=self.secure
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
            adhoc=adhoc,
            secure=self.secure
        )

    def get_internal_listener(self) -> Union[None, ConnectorInfo]:
        """
        Try to get an internal listener.
        """
        return self._get_connector(
            url="",
            active=False,
            internal=True,
            adhoc=False,
            secure=False
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
            adhoc=False,
            secure=False
        )

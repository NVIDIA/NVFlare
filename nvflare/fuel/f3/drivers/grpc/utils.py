# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import grpc

from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.drivers.driver_params import DriverParams


def use_aio_grpc():
    configurator = CommConfigurator()
    return configurator.use_aio_grpc(default=False)


def get_grpc_client_credentials(params: dict):
    root_cert = _read_file(params.get(DriverParams.CA_CERT.value))
    cert_chain = _read_file(params.get(DriverParams.CLIENT_CERT))
    private_key = _read_file(params.get(DriverParams.CLIENT_KEY))
    return grpc.ssl_channel_credentials(
        certificate_chain=cert_chain, private_key=private_key, root_certificates=root_cert
    )


def get_grpc_server_credentials(params: dict):
    root_cert = _read_file(params.get(DriverParams.CA_CERT.value))
    cert_chain = _read_file(params.get(DriverParams.SERVER_CERT))
    private_key = _read_file(params.get(DriverParams.SERVER_KEY))

    return grpc.ssl_server_credentials(
        [(private_key, cert_chain)],
        root_certificates=root_cert,
        require_client_auth=True,
    )


def _read_file(file_name: str):
    if not file_name:
        return None

    with open(file_name, "rb") as f:
        return f.read()

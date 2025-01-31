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

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.drivers.driver_params import DriverParams


def use_aio_grpc():
    configurator = CommConfigurator()
    return configurator.use_aio_grpc(default=False)


def get_grpc_client_credentials(params: dict):
    conn_security = params.get(DriverParams.CONNECTION_SECURITY.value, ConnectionSecurity.MTLS)
    if conn_security == ConnectionSecurity.TLS:
        # One-way SSL
        # For one-way SSL, only CA cert is needed, and no need for client cert and key.
        # We try to use custom CA cert if it's provided. This is because the client may connect to ALB or proxy
        # that provides its CA cert to the client.
        # If the custom CA cert is not provided, we'll use Flare provisioned CA cert.
        params[DriverParams.IMPLEMENTED_CONN_SEC] = "Client TLS: Custom CA Cert used"
        root_cert_file = params.get(DriverParams.CUSTOM_CA_CERT)
        if not root_cert_file:
            params[DriverParams.IMPLEMENTED_CONN_SEC] = "Client TLS: Flare CA Cert used"
            root_cert_file = params.get(DriverParams.CA_CERT.value)
        if not root_cert_file:
            raise RuntimeError(f"cannot get CA cert for one-way SSL: {params}")
        root_cert = _read_file(root_cert_file)
        return grpc.ssl_channel_credentials(root_certificates=root_cert)
    else:
        # For two-way SSL, we always use our own provisioned certs.
        # In the future, we may change to also support other ways to get cert and key.
        params[DriverParams.IMPLEMENTED_CONN_SEC] = "Client mTLS: Flare credentials used"
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

    conn_security = params.get(DriverParams.CONNECTION_SECURITY.value, ConnectionSecurity.MTLS)
    require_client_auth = False if conn_security == ConnectionSecurity.TLS else True

    if require_client_auth:
        params[DriverParams.IMPLEMENTED_CONN_SEC] = "Server mTLS: client auth required"
    else:
        params[DriverParams.IMPLEMENTED_CONN_SEC] = "Server TLS: client auth not required"

    return grpc.ssl_server_credentials(
        [(private_key, cert_chain)],
        root_certificates=root_cert,
        require_client_auth=require_client_auth,
    )


def _read_file(file_name: str):
    if not file_name:
        return None

    with open(file_name, "rb") as f:
        return f.read()

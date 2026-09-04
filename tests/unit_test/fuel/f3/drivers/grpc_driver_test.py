# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography import x509

from nvflare.fuel.f3.drivers import aio_grpc_driver, grpc_driver
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.net_utils import JOB_ID_EXTENSION_OID
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert


def _cert_pem(common_name: str, job_id=None) -> bytes:
    key, pub_key = generate_keys()
    extensions = None
    if job_id:
        extensions = [(x509.UnrecognizedExtension(JOB_ID_EXTENSION_OID, job_id.encode("utf-8")), False)]
    cert = generate_cert(Identity(common_name), Identity(common_name), key, pub_key, extra_extensions=extensions)
    return serialize_cert(cert)


def _authenticated_context(cert_pem: bytes):
    context = MagicMock()
    context.peer.return_value = "ipv4:127.0.0.1:50001"
    context.auth_context.return_value = {"x509_common_name": [b"site-1"], "x509_pem_cert": [cert_pem]}
    return context


def _server():
    server = MagicMock()
    server.connector.params = {DriverParams.HOST.value: "localhost", DriverParams.PORT.value: "8002"}
    return server


@pytest.mark.parametrize("job_id", ["job-1", None])
def test_grpc_servicer_records_peer_identity_of_authenticated_stream(job_id):
    servicer = grpc_driver.Servicer(_server())

    with (
        patch.object(grpc_driver, "StreamConnection") as connection_cls,
        patch.object(grpc_driver.threading, "Thread"),
    ):
        connection_cls.return_value.generate_output.return_value = iter([])
        list(servicer.Stream(iter([]), _authenticated_context(_cert_pem("site-1", job_id))))

    conn_props = connection_cls.call_args.args[2]
    assert conn_props[DriverParams.PEER_CN.value] == "site-1"
    assert conn_props.get(DriverParams.PEER_JOB_ID.value) == job_id


@pytest.mark.parametrize("job_id", ["job-1", None])
def test_aio_grpc_servicer_records_peer_identity_of_authenticated_stream(job_id):
    servicer = aio_grpc_driver.Servicer(_server(), aio_ctx=MagicMock())

    async def consume():
        async for _ in servicer.Stream(iter([]), _authenticated_context(_cert_pem("site-1", job_id))):
            pass

    with patch.object(aio_grpc_driver, "AioStreamSession") as session_cls:
        session_cls.return_value.read_oq = AsyncMock(side_effect=asyncio.CancelledError())
        asyncio.run(consume())

    conn_props = session_cls.call_args.kwargs["conn_props"]
    assert conn_props[DriverParams.PEER_CN.value] == "site-1"
    assert conn_props.get(DriverParams.PEER_JOB_ID.value) == job_id

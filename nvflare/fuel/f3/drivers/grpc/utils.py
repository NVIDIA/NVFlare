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
import asyncio
import base64
import ipaddress
import os
import socket
import ssl
import threading
import time
from urllib.parse import unquote, urlsplit

import grpc
from cryptography import x509

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.drivers.driver_params import DriverParams


def use_aio_grpc():
    configurator = CommConfigurator()
    return configurator.use_aio_grpc(default=None)


class PendingProbeSockets:
    """Make a synchronous TLS probe interruptible while its connector stops."""

    def __init__(self):
        self.lock = threading.Lock()
        self.sockets = set()
        self.closed = False
        self.stopped = threading.Event()
        self.resolver_job = None

    def resolve_addresses(self, host, port, timeout):
        # getaddrinfo has no cancellation API. Keep at most one daemon lookup
        # per connector, and let shutdown release its waiting connector thread.
        with self.lock:
            if self.closed:
                raise ConnectionAbortedError("gRPC certificate probe was stopped")
            job = self.resolver_job
            if job and job["done"].is_set():
                job = None
                self.resolver_job = None
            if job and job["target"] != (host, port):
                raise OSError("another gRPC address lookup is in progress")
            if job is None:
                job = {"target": (host, port), "done": threading.Event()}
                self.resolver_job = job

                def lookup():
                    try:
                        job["addresses"] = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
                    except Exception as ex:
                        job["error"] = ex
                    finally:
                        job["done"].set()

                job["thread"] = threading.Thread(target=lookup, name="grpc-probe-dns", daemon=True)
                job["thread"].start()

        deadline = None if timeout is None else time.monotonic() + timeout
        while not job["done"].wait(0.05):
            if self.stopped.is_set():
                raise ConnectionAbortedError("gRPC certificate probe was stopped")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("gRPC address lookup timed out")
        if self.stopped.is_set():
            raise ConnectionAbortedError("gRPC certificate probe was stopped")
        if "error" in job:
            raise job["error"]
        return job["addresses"]

    def register(self, sock, replace=None):
        with self.lock:
            if replace is not None:
                self.sockets.discard(replace)
            if self.closed:
                sock.close()
                raise ConnectionAbortedError("gRPC certificate probe was stopped")
            self.sockets.add(sock)

    def release(self, sock):
        with self.lock:
            self.sockets.discard(sock)

    def cancel(self):
        with self.lock:
            self.closed = True
            self.stopped.set()
            sockets = tuple(self.sockets)
        for sock in sockets:
            try:
                socket.socket.shutdown(sock, socket.SHUT_RDWR)
            except OSError:
                sock.close()


def _proxy_for_target(host, port, options):
    opts = dict(options or ())
    if opts.get("grpc.enable_http_proxy") in (0, "0", False):
        return None
    bypass = os.environ["no_grpc_proxy"] if "no_grpc_proxy" in os.environ else os.environ.get("no_proxy", "")
    for entry in bypass.split(","):
        entry = entry.strip().lower()
        target = host.lower()
        if not entry:
            continue
        if entry == "*" or entry == target or entry == f"{target}:{port}":
            return None
        if entry.startswith(".") and (target == entry[1:] or target.endswith(entry)):
            return None
        if target.endswith("." + entry):
            return None
        try:
            if ipaddress.ip_address(host) in ipaddress.ip_network(entry, strict=False):
                return None
        except ValueError:
            pass

    if "grpc.http_proxy" in opts:
        proxy = opts["grpc.http_proxy"]
    else:
        proxy = None
        for name in ("grpc_proxy", "https_proxy", "http_proxy"):
            if name in os.environ:
                proxy = os.environ[name]
                break
    if not proxy:
        return None
    if isinstance(proxy, bytes):
        proxy = proxy.decode("utf-8")
    parsed = urlsplit(proxy if "://" in proxy else f"http://{proxy}")
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("unsupported gRPC HTTP proxy configuration")
    auth = None
    if parsed.username is not None:
        user = unquote(parsed.username)
        password = unquote(parsed.password or "")
        auth = base64.b64encode(f"{user}:{password}".encode("utf-8"))
    return parsed.hostname, parsed.port or 80, auth


def _proxy_request(host, port, auth):
    target = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    request = f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n".encode("ascii")
    if auth:
        request += b"Proxy-Authorization: Basic " + auth + b"\r\n"
    return request + b"\r\n"


def _check_proxy_response(response):
    status = response.split(b"\r\n", 1)[0].split()
    if len(status) < 2 or status[0] not in (b"HTTP/1.0", b"HTTP/1.1") or status[1] != b"200":
        raise OSError("gRPC HTTP proxy did not establish a tunnel")


def _connect_raw(host, port, timeout, pending):
    last_error = None
    deadline = None if timeout is None else time.monotonic() + timeout
    for family, socktype, proto, _, address in pending.resolve_addresses(host, port, timeout):
        remaining = None if deadline is None else max(0.001, deadline - time.monotonic())
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("gRPC certificate probe timed out")
        raw = socket.socket(family, socktype, proto)
        raw.settimeout(remaining)
        try:
            pending.register(raw)
            raw.connect(address)
            if deadline is not None:
                raw.settimeout(max(0.001, deadline - time.monotonic()))
            return raw
        except OSError as ex:
            last_error = ex
            pending.release(raw)
            raw.close()
            if pending.closed:
                break
    raise last_error or OSError("gRPC listener address was not found")


def _probe_config(params: dict, options):
    host = params[DriverParams.HOST.value].strip("[]")
    port = int(params[DriverParams.PORT.value])
    opts = dict(options or ())
    target_name = opts.get("grpc.ssl_target_name_override")
    if not target_name:
        authority = opts.get("grpc.default_authority")
        if isinstance(authority, bytes):
            authority = authority.decode("utf-8")
        target_name = urlsplit(f"//{authority}").hostname if authority else host
    if isinstance(target_name, bytes):
        target_name = target_name.decode("utf-8")

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_verify_locations(cafile=params[DriverParams.CA_CERT.value])
    context.load_cert_chain(
        certfile=params[DriverParams.CLIENT_CERT.value], keyfile=params[DriverParams.CLIENT_KEY.value]
    )
    context.set_alpn_protocols(["h2"])

    timeout = params.get(DriverParams.CONNECT_TIMEOUT.value, 60.0)
    return (
        context,
        host,
        port,
        target_name,
        None if timeout is None else float(timeout),
        _proxy_for_target(host, port, options),
    )


def _verified_peer_cert(tls) -> bytes:
    if tls.selected_alpn_protocol() != "h2":
        raise ssl.SSLError("gRPC listener did not negotiate h2")
    peer_cert = tls.getpeercert(binary_form=True)
    if not peer_cert:
        raise ssl.SSLError("gRPC listener did not present a certificate")
    cert = x509.load_der_x509_certificate(peer_cert)
    try:
        constraints = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
    except x509.ExtensionNotFound:
        constraints = None
    if constraints and constraints.ca:
        raise ssl.SSLError("gRPC listener certificate must not be a CA")
    return peer_cert


def probe_grpc_peer_cert(params: dict, options=None, pending: PendingProbeSockets = None) -> bytes:
    """Verify the mTLS listener and return its leaf certificate for channel pinning.

    Python gRPC does not expose the server certificate of a client-side RPC. A
    separate TLS handshake is safe only if the subsequent gRPC channel trusts
    exactly the certificate observed here, rather than the whole CA again.
    """
    context, host, port, target_name, timeout, proxy = _probe_config(params, options)
    pending = pending or PendingProbeSockets()
    raw = _connect_raw(proxy[0] if proxy else host, proxy[1] if proxy else port, timeout, pending)
    active = raw
    try:
        if proxy:
            raw.sendall(_proxy_request(host, port, proxy[2]))
            response = b""
            while b"\r\n\r\n" not in response and len(response) < 8192:
                chunk = raw.recv(8192 - len(response))
                if not chunk:
                    break
                response += chunk
            _check_proxy_response(response)
        tls = context.wrap_socket(raw, server_hostname=target_name, do_handshake_on_connect=False)
        active = tls
        pending.register(tls, replace=raw)
        tls.do_handshake()
        return _verified_peer_cert(tls)
    finally:
        pending.release(active)
        active.close()


async def probe_grpc_peer_cert_aio(params: dict, options=None) -> bytes:
    context, host, port, target_name, timeout, proxy = _probe_config(params, options)
    if proxy:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(proxy[0], proxy[1], limit=8192), timeout=timeout
        )
    else:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                host, port, ssl=context, server_hostname=target_name, ssl_handshake_timeout=timeout
            ),
            timeout=timeout,
        )
    try:
        if proxy:
            writer.write(_proxy_request(host, port, proxy[2]))
            await asyncio.wait_for(writer.drain(), timeout=timeout)
            response = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=timeout)
            _check_proxy_response(response)
            # StreamWriter.start_tls was added in Python 3.11. Upgrade the
            # existing CONNECT stream with the event-loop API available in 3.10.
            writer._transport = await asyncio.wait_for(
                asyncio.get_running_loop().start_tls(
                    writer.transport,
                    writer._protocol,
                    context,
                    server_side=False,
                    server_hostname=target_name,
                    ssl_handshake_timeout=timeout,
                ),
                timeout=timeout,
            )
        return _verified_peer_cert(writer.get_extra_info("ssl_object"))
    finally:
        # The probe never sends application data and HTTP/2 listeners need not
        # answer TLS close_notify. Abort immediately to release the socket on
        # normal completion and cancellation without delaying each connection.
        writer.transport.abort()


def get_grpc_client_credentials(params: dict, peer_cert: bytes = None):
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
        if peer_cert:
            # The probe validated this leaf against the configured CA. Trusting
            # it as the only anchor binds the RPC to the observed certificate.
            root_cert = ssl.DER_cert_to_PEM_cert(peer_cert).encode("ascii")
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

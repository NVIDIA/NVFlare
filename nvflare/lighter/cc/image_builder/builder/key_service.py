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

"""Create-only key service for a Trustee filesystem resource repository.

Run at the trusted KBS administrative boundary. Mount resource storage read-only
in KBS and deny its native write APIs. Revocation state must be stored separately
from restorable resource backups and reconciled before KBS starts. The service
name permits additional key types, while this version accepts only vault keys.
"""

import argparse
import hashlib
import hmac
import http.server
import logging
import os
import socket
import ssl
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

from .common import BuildError, lock, protect_process, read_json, require, validate_resource, write_json

LOG = logging.getLogger(__name__)


def log_failure(operation, error):
    # Do not log exception text: it can contain request data or secret material.
    LOG.error(
        "key_service operation=%s error=%s errno=%s", operation, type(error).__name__, getattr(error, "errno", None)
    )


class BoundedTLSServer(http.server.HTTPServer):
    """Bound admission, handshake and complete request lifetime, including headers."""

    def __init__(self, address, handler_class, tls, *, workers=8, deadline=15):
        self.tls, self.deadline = tls, deadline
        self.slots = threading.BoundedSemaphore(workers)
        super().__init__(address, handler_class)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            threading.Thread(target=self.serve_connection, args=(request, client_address), daemon=True).start()
        except Exception:
            self.slots.release()
            self.shutdown_request(request)
            raise

    def serve_connection(self, request, client_address):
        connection = request
        timer = None

        def expire():
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        try:
            request.settimeout(self.deadline)
            connection = self.tls.wrap_socket(request, server_side=True, do_handshake_on_connect=False)
            timer = threading.Timer(self.deadline, expire)
            timer.daemon = True
            timer.start()
            connection.do_handshake()
            self.finish_request(connection, client_address)
        except (OSError, ValueError) as error:
            log_failure("connection", error)
        finally:
            if timer is not None:
                timer.cancel()
            self.shutdown_request(connection)
            self.slots.release()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ResourceStore:
    def __init__(self, resources, state):
        self.resources, self.state = Path(resources).resolve(), Path(state).resolve()
        require(
            self.resources != self.state
            and self.resources not in self.state.parents
            and self.state not in self.resources.parents,
            "Revocation state must be separate from the resource repository",
        )
        for directory in (self.resources, self.state, self.state / "revoked", self.state / "retired"):
            directory.mkdir(parents=True, mode=0o700, exist_ok=True)

    def _target(self, resource):
        parts = validate_resource(resource)
        # CoCo Trustee v0.22 local_fs stores each key in the repository
        # namespace, escaping path separators in a single filename.
        target = self.resources / resource.replace("/", "\\x2F")
        require(target.resolve().is_relative_to(self.resources), "Resource escapes repository")
        require(
            not any(p.is_symlink() for p in [target, *target.parents] if p != self.resources),
            "Symlink in resource repository",
        )
        return parts, target

    def _tombstone(self, resource):
        return self.state / "revoked" / hashlib.sha256(resource.encode()).hexdigest()

    def put(self, resource, secret):
        require(len(secret) == 64, "Vault secret must be exactly 64 bytes")
        parts, target = self._target(resource)
        # This lock also serializes retirement with activation; no acknowledged
        # create can race the retirement boundary.
        with lock(self.state / "administration.lock"):
            require(
                not (self.state / "retired" / parts[1]).exists() and not self._tombstone(resource).exists(),
                "Identity is revoked",
            )
            approved = read_json(self.state / "approved-bundles.json")
            require(parts[1] in approved["build_ids"], "Bundle is not enabled for key provisioning")
            if target.exists():
                require(hmac.compare_digest(target.read_bytes(), secret), "Conflicting key already exists")
                return False
            target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            fsync_dir(target.parent.parent)
            fsync_dir(self.resources)
            fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=target.parent)
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(secret)
                    stream.flush()
                    os.fsync(stream.fileno())
                # Atomic no-replace publication: the upstream reader sees either
                # no resource or the complete secret, never a partial file.
                os.link(temporary, target)
                fsync_dir(target.parent)
            finally:
                os.unlink(temporary)
                fsync_dir(target.parent)
            return True

    def revoke(self, resource):
        _, target = self._target(resource)
        with lock(self.state / "administration.lock"):
            # Persist denial before removing material; a retry cannot resurrect it.
            write_json(self._tombstone(resource), {"resource": resource})
            target.unlink(missing_ok=True)
            if target.parent.exists():
                fsync_dir(target.parent)

    def reconcile(self):
        """Must run with KBS stopped before serving a restored resource backup."""
        with lock(self.state / "administration.lock"):
            for tombstone in (self.state / "revoked").iterdir():
                _, target = self._target(read_json(tombstone)["resource"])
                target.unlink(missing_ok=True)
                if target.parent.exists():
                    fsync_dir(target.parent)
            for retired in (self.state / "retired").iterdir():
                prefix = "keys\\x2F" + retired.name + "\\x2F"
                for path in self.resources.iterdir():
                    if path.name.startswith(prefix):
                        require(path.is_file() and not path.is_symlink(), "Unexpected resource file")
                        path.unlink()
                fsync_dir(self.resources)

    def retire(self, build_id):
        from .common import identifier

        identifier(build_id)
        with lock(self.state / "administration.lock"):
            write_json(self.state / "retired" / build_id, {"build_id": build_id})
            approved = read_json(self.state / "approved-bundles.json")
            approved["build_ids"] = [value for value in approved["build_ids"] if value != build_id]
            write_json(self.state / "approved-bundles.json", approved)
        self.reconcile()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise BuildError("Administrative redirects are forbidden")


def request(config, method, resource, secret=None):
    validate_resource(resource)
    context = ssl.create_default_context(cafile=config["ca"])
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(config["cert"], config["key"])
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context), NoRedirect())
    req = urllib.request.Request(
        config["url"].rstrip("/") + "/v1/resources/" + resource,
        method=method,
        data=secret,
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with opener.open(req, timeout=30) as response:
            require(response.status in (200, 201, 204), "Key service did not acknowledge operation")
    except urllib.error.HTTPError as exc:
        exc.close()
        raise BuildError(f"Key service rejected operation (HTTP {exc.code})") from None
    except (OSError, urllib.error.URLError):
        raise BuildError("Key service response uncertain; retry the same path and secret") from None


def handler(store, roles):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def respond(self, status):
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

        def perform(self, method):
            fingerprint = hashlib.sha256(self.connection.getpeercert(binary_form=True)).hexdigest()
            role = roles.get(fingerprint)
            if role not in ("builder", "admin") or (method == "DELETE" and role != "admin"):
                self.respond(403)
                return
            if not self.path.startswith("/v1/resources/") or "Transfer-Encoding" in self.headers:
                self.respond(400)
                return
            try:
                resource = self.path[len("/v1/resources/") :]
                validate_resource(resource)
                if method == "PUT":
                    require(self.headers.get_all("Content-Length") == ["64"], "Invalid secret length")
                    self.connection.settimeout(15)
                    secret = self.rfile.read(64)
                    created = store.put(resource, secret)
                    self.respond(201 if created else 204)
                else:
                    store.revoke(resource)
                    self.respond(204)
            except (BuildError, ValueError):
                self.respond(409)
            except Exception as error:
                log_failure(method, error)
                self.respond(503)

        def do_PUT(self):
            self.perform("PUT")

        def do_DELETE(self):
            self.perform("DELETE")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--reconcile", action="store_true")
    args = parser.parse_args()
    try:
        config = read_json(args.config)
        store = ResourceStore(config["resources"], config["state"])
        if args.reconcile:
            store.reconcile()
            return
        protect_process()
        tls = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH, cafile=config["client_ca"])
        tls.minimum_version = ssl.TLSVersion.TLSv1_3
        tls.verify_mode = ssl.CERT_REQUIRED
        tls.load_cert_chain(config["cert"], config["key"])
        server = BoundedTLSServer(
            (config.get("listen", "127.0.0.1"), config["port"]), handler(store, config["certificate_roles"]), tls
        )
        server.serve_forever()
    except (BuildError, OSError, ValueError, KeyError) as error:
        log_failure("startup", error)
        parser.exit(1, "Key service could not start; check its protected configuration\n")


if __name__ == "__main__":
    main()

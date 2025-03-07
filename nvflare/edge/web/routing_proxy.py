# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import hashlib
import json
import logging
import os
import sys
from typing import Tuple
from urllib.parse import urljoin

import requests
from flask import Flask, Response, jsonify, request

from nvflare.edge.web.models.api_error import ApiError
from nvflare.edge.web.web_server import FilteredJSONProvider

# This is just a random large prime number used as number of buckets
PRIME = 100003

log = logging.getLogger(__name__)
app = Flask(__name__)


class UniformHash:
    """A hash algorithm with uniform distribution. It achieves this with following steps,
    1. Get a hash value using SHA256
    2. Map the hash value to a virtual hash table with a large prime number
    3. Map the virtual bucket to real bucket by using an allocation table

    """

    def __init__(self, num_buckets: int):
        self.num_buckets = num_buckets
        self.num = PRIME // num_buckets
        self.remainder = PRIME % num_buckets

    def get_num_buckets(self) -> int:
        return self.num_buckets

    def hash(self, key: str) -> int:
        # The hash() function changes value every run so SHA256 is used
        sha_bytes = hashlib.sha256(key.encode()).digest()
        sha = int.from_bytes(sha_bytes[:8], "big")
        virtual_hash = sha % PRIME

        start = 0
        for i in range(self.num_buckets):
            # Allocation is virtual hash assigned to each bucket, first few buckets get one more if r is not 0
            allocation = (self.num + 1) if i < self.remainder else self.num
            end = start + allocation
            if start <= virtual_hash < end:
                return i
            start = end

        raise RuntimeError("Logic error")


class LcpMapper:
    def __init__(self):
        self.lcp_list = []

    def add_lcp(self, name: str, url: str):
        self.lcp_list.append((name, url))

    def map(self, device_id: str) -> Tuple[str, str]:
        if not self.lcp_list:
            raise RuntimeError("No LCP is configured")

        uniform_hash = UniformHash(len(self.lcp_list))
        index = uniform_hash.hash(device_id)
        return self.lcp_list[index]

    def load_lcp_map(self, mapping_file: str):
        with open(mapping_file, "r") as f:
            mapping = json.load(f)

        for name, config in mapping.items():
            host = config["host"]
            port = config["port"]
            url = f"http://{host}:{port}"
            self.add_lcp(name, url)


def validate_path(path: str):
    if path not in {"job", "task", "result"}:
        raise ApiError(400, "INVALID_REQUEST", f"Invalid path {path}")

    return path


def validate_content(content: bytes):
    if not content:
        return None

    return content


@app.errorhandler(ApiError)
def handle_api_error(error: ApiError):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


mapper = LcpMapper()


@app.route("/<path:path>", methods=["GET", "POST"])
def routing_proxy(path):
    device_id = request.headers.get("X-Flare-Device-ID")
    if not device_id:
        raise ApiError(400, "INVALID_REQUEST", "Device ID is missing")

    name_url = mapper.map(device_id)
    if not name_url:
        raise ApiError(500, "CONFIG_ERROR", f"No LCP configured for device ID {device_id}")

    site_name, url = name_url
    log.info(f"Routing request from device: {device_id} to site {site_name} at {url}")
    validate_path(path)
    target_url = urljoin(url, path)

    try:
        # Prepare headers (remove 'Host' to avoid conflicts)
        headers = {key: value for key, value in request.headers if key.lower() != "host"}

        # Get data from the original request
        data = request.get_data()

        # Forward the request to the target URL
        resp = requests.request(
            method=request.method,
            url=target_url,
            params=request.args,
            headers=headers,
            data=data,
            cookies=request.cookies,
            allow_redirects=False,  # Do not follow redirects
        )

        # Exclude specific headers from the target response
        excluded_headers = ["server", "date", "content-encoding", "content-length", "transfer-encoding", "connection"]
        headers = {name: value for name, value in resp.headers.items() if name.lower() not in excluded_headers}
        headers["Via"] = "edge-proxy"

        # Build the Flask response

        response = Response(validate_content(resp.content), resp.status_code, headers)
        return response

    except requests.exceptions.RequestException as ex:
        raise ApiError(500, "PROXY_ERROR", f"Proxy request failed: {str(ex)}", ex)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <port> <mapping_file>")
        sys.exit(1)

    mapper.load_lcp_map(sys.argv[2])
    proxy_port = int(sys.argv[1])

    app.json = FilteredJSONProvider(app)
    app.run(host="0.0.0.0", port=proxy_port, debug=False)

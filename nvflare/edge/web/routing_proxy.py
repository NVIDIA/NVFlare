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
from urllib.parse import urljoin

from flask import Flask, request, Response, jsonify
import requests

from nvflare.edge.web.models.api_error import ApiError
from nvflare.edge.web.web_server import FilteredJSONProvider

app = Flask(__name__)
lcp_list = [
    "http://localhost:8101",
    "http://localhost:8101"
]


def find_lcp_url(device_id: str) -> str:
    index = hash(device_id) % len(lcp_list)
    return lcp_list[index]


@app.errorhandler(ApiError)
def handle_api_error(error: ApiError):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/<path:path>', methods=['GET', 'POST'])
def routing_proxy(path):

    device_id = request.headers.get("X-Flare-Device-ID")
    if not device_id:
        raise ApiError(400, "INVALID_REQUEST", "Device ID is missing")

    target_url = find_lcp_url(device_id)
    if not target_url:
        raise ApiError(500, "CONFIG_ERROR", f"No LCP configured for device ID {device_id}")

    target_url = urljoin(target_url, path)

    try:
        # Prepare headers (remove 'Host' to avoid conflicts)
        headers = {key: value for key, value in request.headers if key.lower() != 'host'}

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
            allow_redirects=False  # Do not follow redirects
        )

        # Exclude specific headers from the target response
        excluded_headers = ['server', 'date', 'content-encoding', 'content-length', 'transfer-encoding', 'connection']
        headers = {name: value for name, value in resp.headers.items() if name.lower() not in excluded_headers}
        headers["Via"] = "edge-proxy"

        # Build the Flask response
        response = Response(resp.content, resp.status_code, headers)
        return response

    except requests.exceptions.RequestException as ex:
        raise ApiError(500, "PROXY_ERROR", f"Proxy request failed: {str(ex)}", ex)


if __name__ == '__main__':
    app.json = FilteredJSONProvider(app)
    app.run(host='0.0.0.0', port=4321, debug=True)

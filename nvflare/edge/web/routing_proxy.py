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
import logging
import os
import sys
from typing import Any

from flask import Flask, jsonify
from flask.json.provider import DefaultJSONProvider

from nvflare.edge.web.models.api_error import ApiError
from nvflare.edge.web.views.feg_views import api_query, feg_bp

log = logging.getLogger(__name__)
app = Flask(__name__)


def clean_dict(value: Any):
    if isinstance(value, dict):
        return {k: clean_dict(v) for k, v in value.items() if v is not None}
    return value


class FilteredJSONProvider(DefaultJSONProvider):
    sort_keys = False

    def dumps(self, obj: Any, **kwargs: Any) -> str:
        return super().dumps(clean_dict(obj))


@app.errorhandler(ApiError)
def handle_api_error(error: ApiError):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    if len(sys.argv) != 4:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <port> <mapping_file> <ca_cert_file>")
        sys.exit(1)

    proxy_port = int(sys.argv[1])
    lcp_mapping_file = sys.argv[2]
    ca_cert_file = sys.argv[3]

    api_query.set_lcp_mapping(lcp_mapping_file)
    api_query.set_ca_cert(ca_cert_file)
    api_query.start()

    app.json = FilteredJSONProvider(app)
    app.register_blueprint(feg_bp)
    app.run(host="0.0.0.0", port=proxy_port, debug=False)

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
import json
import os

from flask import Flask, jsonify, request, Response

from nvflare.edge.web.models.study_request import StudyRequest
from nvflare.edge.web.models.study_response import StudyResponse

app = Flask("__name__")


@app.route('/study', methods=['POST'])
def study():
    data = request.get_json()
    req = StudyRequest(**data)
    reply = StudyResponse(status="OK",
                          session_id="SessionID-123",
                          study_id="study3",
                          job_id="job123",
                          device_state=req.device_state)

    json_str = json.dumps(reply)
    return Response(json_str, content_type="application/json")

    return jsonify(reply, status=200, mimetype='application/json')


def run_server():
    os.environ["FLASK_ENV"] = "xyz"
    app.run(port=4321, debug=True)


# driver function
if __name__ == '__main__':
    run_server()
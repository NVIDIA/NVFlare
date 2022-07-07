# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
from datetime import datetime

from flask import jsonify, request, send_from_directory

from nvflare.web.app import app
from nvflare.web.models import create_client, get_clients, get_client, patch_client, delete_client, init_db

privilege_dict = {"admin@nvidia.com": "super"}


@app.route("/api/v1/init", methods=["GET"])
def init():
    init_db()
    return jsonify({})


@app.route("/api/v1/clients", methods=["POST"])
def create_one_client():
    req = request.json
    client = create_client(req)
    return jsonify(client)


@app.route("/api/v1/clients", methods=["GET"])
def get_all_clients():
    client_list = get_clients()
    return jsonify(client_list)


@app.route("/api/v1/clients/<id>", methods=["GET", "PATCH", "DELETE"])
def client(id):
    if request.method == "GET":
        client = get_client(id)
        return jsonify(client)
    elif request.method == "PATCH":
        req = request.json
        client = patch_client(id, req)
        return jsonify(client)
    elif request.method == "DELETE":
        delete_client(id)
        return jsonify(client)


@app.route("/api/v1/clients/<id>/blob", methods=["GET"])
def client_blob(id):
    return send_from_directory(directory=os.getcwd(), path="hello.zip", as_attachment=True)


if __name__ == "__main__":
    app.run()

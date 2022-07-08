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
from nvflare.web.models import Store

privilege_dict = {"admin@nvidia.com": "super"}

store = Store()


@app.route("/api/v1/init", methods=["GET"])
def init():
    store.init_db()
    return jsonify({"status": "ok"})


@app.route("/api/v1/clients", methods=["POST"])
def create_one_client():
    req = request.json
    result = store.create_client(req)
    return jsonify(result)


@app.route("/api/v1/clients", methods=["GET"])
def get_all_clients():
    result = store.get_clients()
    return jsonify(result)


@app.route("/api/v1/clients/<id>", methods=["GET", "PATCH", "DELETE"])
def update_client(id):
    if request.method == "GET":
        result = store.get_client(id)
    elif request.method == "PATCH":
        req = request.json
        result = store.patch_client(id, req)
    elif request.method == "DELETE":
        result = store.delete_client(id)
    else:
        result = {"status": "error"}
    return jsonify(result)


@app.route("/api/v1/clients/<id>/blob", methods=["GET"])
def client_blob(id):
    return send_from_directory(directory=os.getcwd(), path="hello.zip", as_attachment=True)


@app.route("/api/v1/users", methods=["POST"])
def create_one_user():
    req = request.json
    result = store.create_user(req)
    return jsonify(result)


@app.route("/api/v1/users", methods=["GET"])
def get_all_users():
    result = store.get_users()
    return jsonify(result)


@app.route("/api/v1/users/<id>", methods=["GET", "PATCH", "DELETE"])
def update_user(id):
    if request.method == "GET":
        result = store.get_user(id)
    elif request.method == "PATCH":
        req = request.json
        result = store.patch_user(id, req)
    elif request.method == "DELETE":
        result = store.delete_user(id)
    else:
        result = {"status": "error"}
    return jsonify(result)


@app.route("/api/v1/users/<id>/auth", methods=["POST"])
def auth_user(id):
    pw = request.json.get("password")
    result = store.auth_user(id, pw)
    return jsonify(result)


@app.route("/api/v1/users/<id>/blob", methods=["GET"])
def user_blob(id):
    return send_from_directory(directory=os.getcwd(), path="hello.zip", as_attachment=True)


if __name__ == "__main__":
    app.run()

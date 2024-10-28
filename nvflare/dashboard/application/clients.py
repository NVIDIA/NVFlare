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

from flask import current_app as app
from flask import jsonify, make_response, request
from flask_jwt_extended import get_jwt, get_jwt_identity, jwt_required

from .store import Store, check_role


@app.route("/api/v1/clients", methods=["POST"])
@jwt_required()
def create_one_client():
    creator = get_jwt_identity()
    req = request.json
    result = Store.create_client(req, creator)
    if result is not None:
        return jsonify(result), 201
    else:
        return jsonify({"status": "conflicting"}), 409


@app.route("/api/v1/clients", methods=["GET"])
@jwt_required()
def get_all_clients():
    result = Store.get_clients()
    return jsonify(result)


@app.route("/api/v1/clients/<id>", methods=["GET"])
@jwt_required()
def get_one_client(id):
    result = Store.get_client(id)
    return jsonify(result)


@app.route("/api/v1/clients/<id>", methods=["PATCH", "DELETE"])
@jwt_required()
def update_client(id):
    creator_id = Store.get_creator_id_by_client_id(id)
    if creator_id is None:
        return jsonify({"status": "error"}), 404
    c, p = check_role(creator_id, get_jwt(), get_jwt_identity())
    if not c and not p:
        return jsonify({"status": "unauthorized"}), 403

    if request.method == "PATCH":
        req = request.json
        if p:
            result = Store.patch_client_by_project_admin(id, req)
        elif c:
            result = Store.patch_client_by_creator(id, req)
    elif request.method == "DELETE":
        result = Store.delete_client(id)
    else:
        result = {"status": "error"}
    if result is not None:
        return jsonify(result)
    else:
        return jsonify({"status": "conflicting"}), 409


@app.route("/api/v1/clients/<int:id>/blob", methods=["POST"])
@jwt_required()
def client_blob(id):
    if not Store._is_approved_by_client_id(id):
        return jsonify({"status": "not approved yet"}), 200
    creator_id = Store.get_creator_id_by_client_id(id)
    c, p = check_role(creator_id, get_jwt(), get_jwt_identity())
    if p or c:
        pin = request.json.get("pin")
        fileobj, filename = Store.get_client_blob(pin, id)
        response = make_response(fileobj.read())
        response.headers.set("Content-Type", "zip")
        response.headers.set("Content-Disposition", f'attachment; filename="{filename}"')
        return response
    else:
        return jsonify({"status": "unauthorized"}), 403

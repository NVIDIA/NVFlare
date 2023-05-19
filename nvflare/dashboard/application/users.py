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


@app.route("/api/v1/users", methods=["POST"])
def create_one_user():
    req = request.json
    result = Store.create_user(req)
    if result is not None:
        return jsonify(result), 201
    else:
        return jsonify({"status": "conflicting"}), 409


@app.route("/api/v1/users", methods=["GET"])
@jwt_required()
def get_all_users():
    claims = get_jwt()
    if claims.get("role") == "project_admin":
        result = Store.get_users()
    else:
        org_name = claims.get("organization", "")
        result = Store.get_users(org_name=org_name)
    return jsonify(result)


@app.route("/api/v1/users/<id>", methods=["GET"])
@jwt_required()
def get_one_user(id):
    c, p = check_role(id, get_jwt(), get_jwt_identity())
    if not c and not p:
        return jsonify({"status": "unauthorized"}), 403

    return jsonify(Store.get_user(id))


@app.route("/api/v1/users/<id>", methods=["PATCH", "DELETE"])
@jwt_required()
def update_user(id):
    c, p = check_role(id, get_jwt(), get_jwt_identity())
    if not c and not p:
        return jsonify({"status": "unauthorized"}), 403

    if request.method == "PATCH":
        req = request.json
        req.pop("email", None)
        if p:
            result = Store.patch_user_by_project_admin(id, req)
        elif c:
            result = Store.patch_user_by_creator(id, req)
    elif request.method == "DELETE":
        result = Store.delete_user(id)
    else:
        result = {"status": "error"}
    return jsonify(result)


@app.route("/api/v1/users/<int:id>/blob", methods=["POST"])
@jwt_required()
def user_blob(id):
    if not Store._is_approved_by_user_id(id):
        return jsonify({"status": "not approved yet"}), 200
    c, p = check_role(id, get_jwt(), get_jwt_identity())
    if not c and not p:
        return jsonify({"status": "unauthorized"}), 403
    pin = request.json.get("pin")
    fileobj, filename = Store.get_user_blob(pin, id)
    response = make_response(fileobj.read())
    response.headers.set("Content-Type", "zip")
    response.headers.set("Content-Disposition", f'attachment; filename="{filename}"')
    return response

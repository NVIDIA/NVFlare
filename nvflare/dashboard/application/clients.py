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
from flask_jwt_extended import get_jwt_identity, jwt_required

from nvflare.dashboard.application.constants import FLARE_DASHBOARD_NAMESPACE

from .blob import gen_client_blob
from .models import Client, User
from .store import Store, is_project_admin
from .validation import DashboardError


def _owned_client(id):
    user = Store.get_requester(get_jwt_identity())
    client = Client.query.get(id)
    if client is None:
        raise DashboardError("Client site not found.", status_code=404)
    Store.require_site_owner(user, client)
    return user, client


@app.route(FLARE_DASHBOARD_NAMESPACE + "/api/v1/clients", methods=["POST"])
@jwt_required()
def create_one_client():
    creator = get_jwt_identity()
    req = request.json
    if "props" in req:
        return jsonify({"status": "error", "message": "Client props cannot be supplied through the Dashboard API"}), 400
    result = Store.create_client(req, creator)
    if result is not None:
        return jsonify(result), 201
    else:
        return jsonify({"status": "conflicting"}), 409


@app.route(FLARE_DASHBOARD_NAMESPACE + "/api/v1/clients", methods=["GET"])
@jwt_required()
def get_all_clients():
    user = Store.get_requester(get_jwt_identity())
    if not is_project_admin(user) and user.role.name != "org_admin":
        return jsonify(status="ok", client_list=[])
    Store.require_site_owner(user)
    result = (
        Store.get_clients()
        if is_project_admin(user)
        else Store.get_clients(org=user.organization.name, creator_id=user.id, pending_only=user.approval_state < 100)
    )
    return jsonify(result)


@app.route(FLARE_DASHBOARD_NAMESPACE + "/api/v1/clients/<id>", methods=["GET"])
@jwt_required()
def get_one_client(id):
    _, client = _owned_client(id)
    return jsonify(status="ok", client=client.asdict())


@app.route(FLARE_DASHBOARD_NAMESPACE + "/api/v1/clients/<id>", methods=["PATCH", "DELETE"])
@jwt_required()
def update_client(id):
    user, client = _owned_client(id)
    p = is_project_admin(user)

    if request.method == "PATCH":
        req = request.json
        if "props" in req:
            return (
                jsonify({"status": "error", "message": "Client props cannot be supplied through the Dashboard API"}),
                400,
            )
        if not p and "approval_state" in req:
            raise DashboardError("Only the project administrator can approve or deny client sites.")
        if not p and client.approval_state >= 100 and req.get("name", client.name) != client.name:
            raise DashboardError("Only the project administrator can rename an approved client site.")
        if p:
            result = Store.patch_client_by_project_admin(id, req)
        else:
            result = Store.patch_client_by_creator(id, req)
    elif request.method == "DELETE":
        if not p and client.approval_state >= 100:
            raise DashboardError("Only the project administrator can delete an approved client site.")
        result = Store.delete_client(id, creator_id=None if p else user.id)
    else:
        result = {"status": "error"}
    if result is not None:
        return jsonify(result)
    else:
        return jsonify({"status": "conflicting"}), 409


@app.route(FLARE_DASHBOARD_NAMESPACE + "/api/v1/clients/<int:id>/blob", methods=["POST"])
@jwt_required()
def client_blob(id):
    user, client = _owned_client(id)
    owner = User.query.get(client.creator_id)
    if user.approval_state < 100 or owner is None or owner.approval_state < 100 or client.approval_state < 100:
        raise DashboardError(
            "The project administrator must approve both the site owner and the client site before download."
        )
    pin = request.json.get("pin")
    fileobj, filename = gen_client_blob(pin, id)
    response = make_response(fileobj.read())
    response.headers.set("Content-Type", "zip")
    response.headers.set("Content-Disposition", f'attachment; filename="{filename}"')
    return response

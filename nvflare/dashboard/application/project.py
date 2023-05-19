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
from flask_jwt_extended import create_access_token, get_jwt, jwt_required

from . import jwt
from .store import Store


@jwt.expired_token_loader
def my_expired_token_callback(jwt_header, jwt_payload):
    return jsonify({"status": "unauthenticated"}), 401


@app.route("/application-config")
def application_config_html():
    return app.send_static_file("application-config.html")


@app.route("/downloads")
def downloads_html():
    return app.send_static_file("downloads.html")


@app.route("/")
def index_html():
    return app.send_static_file("index.html")


@app.route("/logout")
def logout_html():
    return app.send_static_file("logout.html")


@app.route("/project-admin-dashboard")
def project_admin_dashboard_html():
    return app.send_static_file("project-admin-dashboard.html")


@app.route("/project-configuration")
def project_configuration_html():
    return app.send_static_file("project-configuration.html")


@app.route("/registration-form")
def registration_form_html():
    return app.send_static_file("registration-form.html")


@app.route("/server-config")
def server_config_html():
    return app.send_static_file("server-config.html")


@app.route("/site-dashboard")
def site_dashboard_html():
    return app.send_static_file("site-dashboard.html")


@app.route("/user-dashboard")
def user_dashboard_html():
    return app.send_static_file("user-dashboard.html")


@app.route("/api/v1/login", methods=["POST"])
def login():
    req = request.json
    email = req.get("email", None)
    password = req.get("password", None)
    user = Store.verify_user(email, password)
    if user:
        additional_claims = {"role": user.role.name, "organization": user.organization.name}
        access_token = create_access_token(identity=user.email, additional_claims=additional_claims)
        return jsonify(
            {
                "status": "ok",
                "user": {"id": user.id, "email": user.email, "role": user.role.name},
                "access_token": access_token,
            }
        )
    else:
        return jsonify({"status": "unauthenticated"}), 401


@app.route("/api/v1/overseer/blob", methods=["POST"])
@jwt_required()
def overseer_blob():
    claims = get_jwt()
    if claims.get("role") == "project_admin":
        pin = request.json.get("pin")
        fileobj, filename = Store.get_overseer_blob(pin)
        response = make_response(fileobj.read())
        response.headers.set("Content-Type", "zip")
        response.headers.set("Content-Disposition", f'attachment; filename="{filename}"')
        return response
    else:
        return jsonify({"status": "unauthorized"}), 403


@app.route("/api/v1/servers/<int:id>/blob", methods=["POST"])
@jwt_required()
def server_blob(id):
    claims = get_jwt()
    if claims.get("role") == "project_admin":
        pin = request.json.get("pin")
        fileobj, filename = Store.get_server_blob(pin, id == 1)
        response = make_response(fileobj.read())
        response.headers.set("Content-Type", "zip")
        response.headers.set("Content-Disposition", f'attachment; filename="{filename}"')
        return response
    else:
        return jsonify({"status": "unauthorized"}), 403


@app.route("/api/v1/project", methods=["PATCH"])
@jwt_required()
def set_project():
    claims = get_jwt()
    if claims.get("role") == "project_admin":
        req = request.json
        return jsonify(Store.set_project(req))
    else:
        return jsonify({"status": "unauthorized"}), 403


@app.route("/api/v1/project", methods=["GET"])
def get_project():
    return jsonify(Store.get_project())


@app.route("/api/v1/organizations", methods=["GET"])
def get_orgs():
    return jsonify(Store.get_orgs())

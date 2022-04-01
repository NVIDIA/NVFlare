# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from datetime import datetime

from flask import jsonify, request

from nvflare.ha.overseer.app import app
from nvflare.ha.overseer.utils import (
    get_all_sp,
    get_primary_sp,
    load_privilege,
    promote_sp,
    simple_PSP_policy,
    update_sp_state,
)

privilege_dict = load_privilege()


@app.route("/api/v1/heartbeat", methods=["GET", "POST"])
def heartbeat():
    if request.method == "POST":
        req = request.json
        project = req.get("project")
        role = req.get("role")
        if project is None or role is None:
            return jsonify({"error": "project and role must be provided"})
        now = datetime.utcnow()
        update_sp_state(project, now)
        if role == "server":
            sp_end_point = req.get("sp_end_point")
            if sp_end_point is None:
                return jsonify({"error": "sp_end_point is not provided"})
            incoming_sp = dict(sp_end_point=sp_end_point, project=project)
            psp = simple_PSP_policy(incoming_sp, now)
        elif role in ["client", "admin"]:
            psp = get_primary_sp(project)
        else:
            psp = {}
        return jsonify({"primary_sp": psp, "sp_list": get_all_sp(project)})


@app.route("/api/v1/promote", methods=["GET", "POST"])
def promote():
    if request.headers.get("X-USER") not in privilege_dict.get("super", {}):
        return jsonify({"Error": "No rights"})
    if request.method == "POST":
        req = request.json
        sp_end_point = req.get("sp_end_point", "")
        project = req.get("project", "")
        if project and sp_end_point:
            incoming_sp = dict(sp_end_point=sp_end_point, project=project)
            err, result = promote_sp(incoming_sp)
            if not err:
                return jsonify({"primary_sp": result})
            else:
                return jsonify({"Error": result})
        else:
            return jsonify({"Error": "Wrong project or sp_end_point."})


@app.route("/api/v1/refresh")
def refresh():
    if request.headers.get("X-USER") not in privilege_dict.get("super", {}):
        return jsonify({"Error": "No rights"})
    return jsonify({"Status": "Error.  API disabled."})


if __name__ == "__main__":
    app.run()

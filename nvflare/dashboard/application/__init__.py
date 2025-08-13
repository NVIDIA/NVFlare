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

from flask import Flask
from flask_jwt_extended import JWTManager
from flask_sqlalchemy import SQLAlchemy

from nvflare.dashboard.utils import EnvVar, get_web_root

db = SQLAlchemy()
jwt = JWTManager()


def init_app():
    web_root = get_web_root()
    os.makedirs(web_root, exist_ok=True)
    static_folder = os.environ.get(EnvVar.DASHBOARD_STATIC_FOLDER, "static")
    app = Flask(__name__, static_url_path="", static_folder=static_folder)
    app.config.from_object("nvflare.dashboard.config.Config")
    db.init_app(app)
    jwt.init_app(app)
    with app.app_context():
        from . import clients, project, users
        from .store import Store

        db.create_all()
        if not Store.ready():
            credential = os.environ.get(EnvVar.CREDENTIAL)
            if credential is None:
                print(f"Please set env var {EnvVar.CREDENTIAL}")
                exit(1)
            parts = credential.split(":")
            if len(parts) != 3:
                print(f"Invalid value '{credential}' for env var {EnvVar.CREDENTIAL}: it must be email:password:org")
            email = parts[0]
            pwd = parts[1]
            org = parts[2]
            Store.seed_user(email, pwd, org)
    with open(os.path.join(web_root, ".db_init_done"), "ab") as f:
        f.write(bytes())
    return app

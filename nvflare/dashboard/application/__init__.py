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

db = SQLAlchemy()
jwt = JWTManager()


def init_app():
    os.makedirs("/var/tmp/nvflare/dashboard", exist_ok=True)
    static_folder = os.environ.get("NVFL_DASHBOARD_STATIC_FOLDER", "static")
    app = Flask(__name__, static_url_path="", static_folder=static_folder)
    app.config.from_object("nvflare.dashboard.config.Config")
    db.init_app(app)
    jwt.init_app(app)
    with app.app_context():
        from . import clients, project, users
        from .store import Store

        db.create_all()
        if not Store.ready():
            credential = os.environ.get("NVFL_CREDENTIAL")
            if credential is None:
                print("Please set env var NVFL_CREDENTIAL")
                exit(1)
            email = credential.split(":")[0]
            pwd = credential.split(":")[1]
            Store.seed_user(email, pwd)
    with open(os.path.join("/var/tmp/nvflare/dashboard", ".db_init_done"), "ab") as f:
        f.write(bytes())
    return app

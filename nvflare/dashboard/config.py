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

"""Flask configuration variables."""
import json
import os
from datetime import timedelta

import yaml

from nvflare.dashboard.utils import EnvVar, get_web_root
from nvflare.lighter.utils import generate_password


class Config:
    # General Config
    SECRET_KEY = os.environ.get(EnvVar.SECRET_KEY, generate_password(16))
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=30)

    # Database
    web_root = get_web_root()
    default_sqlite_file = os.path.join(web_root, "db.sqlite")
    default_sqlite_url = f"sqlite:///{default_sqlite_file}"
    SQLALCHEMY_DATABASE_URI = os.environ.get(EnvVar.DATABASE_URL, default_sqlite_url)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False


class PropertyManager:

    def __init__(self):
        web_root = get_web_root()
        self.props = {}
        yml_file = os.path.join(web_root, "properties.yml")
        if os.path.exists(yml_file):
            with open(yml_file, "r") as f:
                self.props = yaml.safe_load(f)
            return

        json_file = os.path.join(web_root, "properties.json")
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                self.props = json.load(f)

    def get_project_props(self):
        return self.props.get("project", {})

    def get_project_prop(self, key, default=None):
        props = self.get_project_props()
        return props.get(key, default)

    def get_client_props(self):
        return self.props.get("client", {})

    def get_client_prop(self, key, default=None):
        props = self.get_client_props()
        return props.get(key, default)

    def get_server_props(self):
        return self.props.get("server", {})

    def get_server_prop(self, key, default=None):
        props = self.get_server_props()
        return props.get(key, default)

    def get_admin_props(self):
        return self.props.get("admin", {})

    def get_admin_prop(self, key, default=None):
        props = self.get_admin_props()
        return props.get(key, default)

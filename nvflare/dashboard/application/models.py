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

import json
from datetime import datetime

from . import db


class CommonMixin(object):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(512), default="")
    description = db.Column(db.String(512), default="")
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def asdict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Organization(CommonMixin, db.Model):
    def asdict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns if c.name in ("name",)}


class Role(CommonMixin, db.Model):
    pass


def _fix_props(obj, table_dict: dict, column: str):
    value = getattr(obj, column)
    if value:
        table_dict[column] = json.loads(value)
    else:
        table_dict.pop(column, None)


class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    frozen = db.Column(db.Boolean, default=False)
    public = db.Column(db.Boolean, default=False)
    short_name = db.Column(db.String(128), default="")
    title = db.Column(db.String(512), default="")
    description = db.Column(db.String(2048), default="")
    app_location = db.Column(db.String(2048), default="")
    ha_mode = db.Column(db.Boolean, default=False)
    starting_date = db.Column(db.String(128), default="")
    end_date = db.Column(db.String(128), default="")
    overseer = db.Column(db.String(128), default="")
    server1 = db.Column(db.String(128), default="")
    server2 = db.Column(db.String(128), default="")
    root_cert = db.Column(db.String(4096), default="")
    root_key = db.Column(db.String(4096), default="")
    cc_mode = db.Column(db.Boolean, default=False)
    project_props = db.Column(db.String(2048), default="")  # additional project properties - JSON string
    server_props = db.Column(db.String(2048), default="")  # additional server properties - JSON string

    def asdict(self):
        table_dict = {
            c.name: getattr(self, c.name)
            for c in self.__table__.columns
            if c.name not in ["id", "root_cert", "root_key"]
        }
        _fix_props(self, table_dict, "project_props")
        _fix_props(self, table_dict, "server_props")

        return table_dict


class Client(CommonMixin, db.Model):
    name = db.Column(db.String(512), unique=True)
    organization_id = db.Column(db.Integer, db.ForeignKey("organization.id"), nullable=False)
    organization = db.relationship("Organization", backref="clients")
    creator_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    approval_state = db.Column(db.Integer, default=0)
    download_count = db.Column(db.Integer, default=0)
    capacity = db.Column(db.String(2048), default="")  # JSON string
    props = db.Column(db.String(2048), default="")  # additional properties - JSON string

    def asdict(self):
        table_dict = {c.name: getattr(self, c.name) for c in self.__table__.columns if "_id" not in c.name}
        table_dict.update(
            {
                "organization": self.organization.name,
            }
        )
        _fix_props(self, table_dict, "props")
        _fix_props(self, table_dict, "capacity")
        return table_dict


class User(CommonMixin, db.Model):
    email = db.Column(db.String(128), unique=True)
    password_hash = db.Column(db.String(128))
    role_id = db.Column(db.Integer, db.ForeignKey("role.id"), nullable=False)
    role = db.relationship("Role", backref="users")
    organization_id = db.Column(db.Integer, db.ForeignKey("organization.id"), nullable=False)
    organization = db.relationship("Organization", backref="users")
    props = db.Column(db.String(1048), default="")  # additional properties - JSON string
    approval_state = db.Column(db.Integer, default=0)
    download_count = db.Column(db.Integer, default=0)

    def asdict(self):
        table_dict = {c.name: getattr(self, c.name) for c in self.__table__.columns if "_id" not in c.name}
        table_dict.update({"organization": self.organization.name, "role": self.role.name})
        _fix_props(self, table_dict, "props")
        table_dict.pop("password_hash")
        return table_dict

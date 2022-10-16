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


class Capacity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    capacity = db.Column(db.String(1024), default="")
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def asdict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


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

    def asdict(self):
        table_dict = {
            c.name: getattr(self, c.name)
            for c in self.__table__.columns
            if c.name not in ["id", "root_cert", "root_key"]
        }
        return table_dict


class Client(CommonMixin, db.Model):
    capacity_id = db.Column(db.Integer, db.ForeignKey("capacity.id"), nullable=False)
    name = db.Column(db.String(512), unique=True)
    organization_id = db.Column(db.Integer, db.ForeignKey("organization.id"), nullable=False)
    capacity = db.relationship("Capacity", backref="clients")
    organization = db.relationship("Organization", backref="clients")
    creator_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    approval_state = db.Column(db.Integer, default=0)
    download_count = db.Column(db.Integer, default=0)

    def asdict(self):
        table_dict = {c.name: getattr(self, c.name) for c in self.__table__.columns if "_id" not in c.name}
        table_dict.update({"organization": self.organization.name, "capacity": json.loads(self.capacity.capacity)})
        return table_dict


class User(CommonMixin, db.Model):
    email = db.Column(db.String(128), unique=True)
    password_hash = db.Column(db.String(128))
    role_id = db.Column(db.Integer, db.ForeignKey("role.id"), nullable=False)
    role = db.relationship("Role", backref="users")
    organization_id = db.Column(db.Integer, db.ForeignKey("organization.id"), nullable=False)
    organization = db.relationship("Organization", backref="users")
    approval_state = db.Column(db.Integer, default=0)
    download_count = db.Column(db.Integer, default=0)

    def asdict(self):
        table_dict = {c.name: getattr(self, c.name) for c in self.__table__.columns if "_id" not in c.name}
        table_dict.update({"organization": self.organization.name, "role": self.role.name})
        table_dict.pop("password_hash")
        return table_dict

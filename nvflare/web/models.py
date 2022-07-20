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

# from .refresh import db

# from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

from .app import app

app.config.from_mapping(
    SECRET_KEY=os.environ.get("SECRET_KEY") or "dev_key",
    SQLALCHEMY_DATABASE_URI=os.environ.get("DATABASE_URL")
    or "sqlite:///" + os.path.join(os.getcwd(), "webserver.sqlite"),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
)
db = SQLAlchemy(app)


class CommonMixin(object):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(512), default="")
    description = db.Column(db.String(512), default="")
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def asdict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Organization(CommonMixin, db.Model):
    pass


class Role(CommonMixin, db.Model):
    pass


class Capacity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    capacity = db.Column(db.String(1024), default="")
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def asdict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Client(CommonMixin, db.Model):
    capacity_id = db.Column(db.Integer, db.ForeignKey("capacity.id"), nullable=False)
    organization_id = db.Column(db.Integer, db.ForeignKey("organization.id"), nullable=False)
    capacity = db.relationship("Capacity", backref="clients")
    organization = db.relationship("Organization", backref="clients")
    approval_state = db.Column(db.Integer, default=0)

    def asdict(self):
        table_dict = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        table_dict.update({"organization": self.organization.name, "capacity": json.loads(self.capacity.capacity)})
        if self.approval_state > 100:
            table_dict.update({"download_url": f"./clients/{self.id}/blob"})
        return table_dict


class User(CommonMixin, db.Model):
    email = db.Column(db.String(128), unique=True)
    password_hash = db.Column(db.String(128))
    role_id = db.Column(db.Integer, db.ForeignKey("role.id"), nullable=False)
    role = db.relationship("Role", backref="users")
    organization_id = db.Column(db.Integer, db.ForeignKey("organization.id"), nullable=False)
    organization = db.relationship("Organization", backref="users")
    approval_state = db.Column(db.Integer, default=0)

    def asdict(self):
        table_dict = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        table_dict.update({"organization": self.organization.name, "role": self.role.name})
        # table_dict.pop("password_hash")
        if self.approval_state > 100:
            table_dict.update({"download_url": f"./users/{self.id}/blob"})
        return table_dict


def _dict_or_empty(item):
    return item.asdict() if item else {}


def get_or_create(session, model, **kwargs):
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance
    else:
        instance = model(**kwargs)
        session.add(instance)
        session.commit()
        return instance


def add_ok(obj):
    obj.update({"status": "ok"})
    return obj


class Store(object):
    def init_db(self):
        db.drop_all()
        db.create_all()
        return add_ok({})

    def create_client(self, req):
        name = req.get("name")
        organization = req.get("organization", "")
        capacity = req.get("capacity")
        description = req.get("description", "")
        org = get_or_create(db.session, Organization, name=organization)
        if capacity is not None:
            cap = get_or_create(db.session, Capacity, capacity=json.dumps(capacity))
        client = Client(name=name, description=description)
        client.organization_id = org.id
        client.capacity_id = cap.id
        db.session.add(client)
        db.session.commit()
        return add_ok({"client": _dict_or_empty(client)})

    def get_clients(self):
        all_clients = Client.query.all()
        return add_ok({"client_list": [_dict_or_empty(client) for client in all_clients]})

    def get_client(self, id):
        client = Client.query.get(id)
        return add_ok({"client": _dict_or_empty(client)})

    def patch_client(self, id, req):
        client = Client.query.get(id)
        organization = req.pop("organization", None)
        if organization is not None:
            org = get_or_create(db.session, Organization, name=organization)
            client.organization_id = org.id
        capacity = req.pop("capacity", None)
        if capacity is not None:
            capacity = json.dumps(capacity)
            cap = get_or_create(db.session, Capacity, capacity=capacity)
            client.capacity_id = cap.id
        for k, v in req.items():
            setattr(client, k, v)
        db.session.add(client)
        db.session.commit()
        return add_ok({"client": _dict_or_empty(client)})

    def delete_client(self, id):
        Client.query.get(id).delete()
        return add_ok({})

    def create_user(self, req):
        name = req.get("name")
        email = req.get("email")
        password = req.get("password", "")
        password_hash = generate_password_hash(password)
        organization = req.get("organization")
        role_name = req.get("role")
        description = req.get("description", "")
        org = get_or_create(db.session, Organization, name=organization)
        role = get_or_create(db.session, Role, name=role_name)
        user = User(email=email, name=name, password_hash=password_hash, description=description)
        user.organization_id = org.id
        user.role_id = role.id
        db.session.add(user)
        db.session.commit()
        return add_ok({"user": _dict_or_empty(user)})

    def get_users(self):
        all_users = User.query.all()
        return add_ok({"user_list": [_dict_or_empty(client) for client in all_users]})

    def get_user(self, id):
        user = User.query.get(id)
        return add_ok({"user": _dict_or_empty(user)})

    def patch_user(self, id, req):
        user = User.query.get(id)
        organization = req.pop("organization", None)
        if organization is not None:
            org = get_or_create(db.session, Organization, name=organization)
            user.organization_id = org.id
        role_name = req.pop("role", None)
        if role_name is not None:
            role = get_or_create(db.session, Role, name=role_name)
            user.role_id = role.id
        password = req.pop("password", None)
        if password is not None:
            password_hash = generate_password_hash(password)
            user.password_hash = password_hash
        for k, v in req.items():
            setattr(user, k, v)
        db.session.add(user)
        db.session.commit()
        return add_ok({"user": _dict_or_empty(user)})

    def auth_user(self, id, pw):
        user = User.query.get(id)
        password_hash = user.password_hash
        result = check_password_hash(password_hash, pw)
        if result:
            return add_ok({})
        else:
            return {"status": "error"}

    def delete_user(self, id):
        User.query.get(id).delete()
        return add_ok({})

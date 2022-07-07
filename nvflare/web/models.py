import os

from flask_sqlalchemy import SQLAlchemy

from .app import app

app.config.from_mapping(
    SECRET_KEY=os.environ.get("SECRET_KEY") or "dev_key",
    SQLALCHEMY_DATABASE_URI=os.environ.get("DATABASE_URL")
    or "sqlite:///" + os.path.join(os.getcwd(), "webserver.sqlite"),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
)
db = SQLAlchemy(app)


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

from .refresh import db

# from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone


class CommonMixin(object):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(25))
    description = db.Column(db.String(256), default="")
    created_at = db.Column(db.Float, nullable=False, default=datetime.now(timezone.utc).timestamp)
    updated_at = db.Column(
        db.Float, default=datetime.now(timezone.utc).timestamp, onupdate=datetime.now(timezone.utc).timestamp
    )


class Organization(CommonMixin, db.Model):
    def asdict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Role(CommonMixin, db.Model):
    def asdict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Resource(CommonMixin, db.Model):
    resource = db.Column(db.String(128), default="")

    def asdict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Client(CommonMixin, db.Model):
    resource_id = db.Column(db.Integer, db.ForeignKey("resource.id"), nullable=False)
    organization_id = db.Column(db.Integer, db.ForeignKey("organization.id"), nullable=False)
    resource = db.relationship("Resource", backref="clients")
    organization = db.relationship("Organization", backref="clients")
    approval_state = db.Column(db.Integer, default=0)

    def asdict(self):
        table_dict = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        table_dict.update({"organization": self.organization.asdict(), "resource": self.resource.asdict()})
        if self.approval_state > 100:
            table_dict.update({"download_url": f"./clients/{self.id}/blob"})
        return table_dict


class User(CommonMixin, db.Model):
    email = db.Column(db.String(128))
    hashed_pw = db.Column(db.String(128))
    role = db.Column(db.Integer, db.ForeignKey("role.id"), nullable=False)
    organization = db.Column(db.Integer, db.ForeignKey("organization.id"), nullable=False)
    approval_state = db.Column(db.Integer, default=0)
    gender = db.Column(db.String(10))


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


def init_db():
    db.drop_all()
    db.create_all()


def create_client(req):
    name = req.get("name")
    organization = req.get("organization")
    resource = req.get("resource")
    description = req.get("description", "")
    org = get_or_create(db.session, Organization, name=organization)
    res = get_or_create(db.session, Resource, resource=resource)
    client = Client(name=name, description=description)
    client.organization_id = org.id
    client.resource_id = res.id
    db.session.add(client)
    db.session.commit()
    return _dict_or_empty(client)


def get_clients():
    all_clients = Client.query.all()
    return [_dict_or_empty(client) for client in all_clients]


def get_client(id):
    client = Client.query.get(id)
    return _dict_or_empty(client)


def patch_client(id, req):
    print(id)
    print(req)
    client = Client.query.get(id)
    for k, v in req.items():
        setattr(client, k, v)
    db.session.add(client)
    db.session.commit()
    return _dict_or_empty(client)


def delete_client(id):
    Client.query.filter_by(id=id).delete()
    return True

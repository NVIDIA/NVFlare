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
import logging

from werkzeug.security import check_password_hash, generate_password_hash

from .blob import gen_client, gen_overseer, gen_server, gen_user
from .cert import Entity, make_root_cert
from .models import Capacity, Client, Organization, Project, Role, User, db

log = logging.getLogger(__name__)


def check_role(id, claims, requester):
    is_creator = requester == Store._get_email_by_id(id)
    is_project_admin = claims.get("role") == "project_admin"
    return is_creator, is_project_admin


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


def inc_dl(model, id):
    instance = model.query.get(id)
    instance.download_count = instance.download_count + 1
    db.session.add(instance)
    db.session.commit()


class Store(object):
    @classmethod
    def ready(cls):
        user = User.query.get(1)
        return user.approval_state >= 100 if user else False

    @classmethod
    def seed_user(cls, email, pwd):
        seed_user = {
            "name": "super_name",
            "email": email,
            "password": pwd,
            "organization": "",
            "role": "project_admin",
            "approval_state": 200,
        }
        cls.create_user(seed_user)
        cls.create_project()
        return email, pwd

    @classmethod
    def init_db(cls):
        db.drop_all()
        db.create_all()
        return add_ok({})

    @classmethod
    def create_project(cls):
        project = Project()
        db.session.add(project)
        db.session.commit()
        return add_ok({"project": _dict_or_empty(project)})

    @classmethod
    def build_project(cls, project):
        entity = Entity(project.short_name)
        cert_pair = make_root_cert(entity)
        project.root_cert = cert_pair.ser_cert
        project.root_key = cert_pair.ser_pri_key
        db.session.add(project)
        db.session.commit()
        return add_ok({"project": _dict_or_empty(project)})

    @classmethod
    def _add_registered_info(cls, project_dict):
        project_dict["num_clients"] = Client.query.count()
        project_dict["num_orgs"] = Organization.query.count()
        project_dict["num_users"] = User.query.count()
        return project_dict

    @classmethod
    def set_project(cls, req):
        project = Project.query.first()
        if project.frozen:
            return {"status": "Project is frozen"}
        req.pop("id", None)
        short_name = req.pop("short_name", "")
        if short_name:
            if len(short_name) > 16:
                short_name = short_name[:16]
            project.short_name = short_name
        for k, v in req.items():
            setattr(project, k, v)
        db.session.add(project)
        db.session.commit()
        if project.frozen:
            cls.build_project(project)
        project_dict = _dict_or_empty(project)
        project_dict = cls._add_registered_info(project_dict)
        return add_ok({"project": project_dict})

    @classmethod
    def get_project(cls):
        project_dict = _dict_or_empty(Project.query.first())
        project_dict = cls._add_registered_info(project_dict)
        return add_ok({"project": project_dict})

    @classmethod
    def get_overseer_blob(cls, key):
        fileobj, filename = gen_overseer(key)
        return fileobj, filename

    @classmethod
    def get_server_blob(cls, key, first_server=True):
        fileobj, filename = gen_server(key, first_server)
        return fileobj, filename

    @classmethod
    def get_orgs(cls):
        all_orgs = Organization.query.all()

        return add_ok({"client_list": [_dict_or_empty(org) for org in all_orgs]})

    @classmethod
    def _is_approved_by_client_id(cls, id):
        client = Client.query.get(id)
        return client.approval_state >= 100

    @classmethod
    def _is_approved_by_user_id(cls, id):
        user = User.query.get(id)
        return user.approval_state >= 100

    @classmethod
    def create_client(cls, req, creator):
        creator_id = User.query.filter_by(email=creator).first().id
        name = req.get("name")
        organization = req.get("organization", "")
        capacity = req.get("capacity")
        description = req.get("description", "")
        org = get_or_create(db.session, Organization, name=organization)
        if capacity is not None:
            cap = get_or_create(db.session, Capacity, capacity=json.dumps(capacity))
        client = Client(name=name, description=description, creator_id=creator_id)
        client.organization_id = org.id
        client.capacity_id = cap.id
        try:
            db.session.add(client)
            db.session.commit()
        except Exception as e:
            log.error(f"Error while creating client: {e}")
            return None
        return add_ok({"client": _dict_or_empty(client)})

    @classmethod
    def get_clients(cls, org=None):
        if org is None:
            all_clients = Client.query.all()
        else:
            all_clients = Organization.query.filter_by(name=org).first().clients

        return add_ok({"client_list": [_dict_or_empty(client) for client in all_clients]})

    @classmethod
    def get_creator_id_by_client_id(cls, id):
        client = Client.query.get(id)
        if client:
            creator_id = client.creator_id
            return creator_id
        else:
            return None

    @classmethod
    def get_client(cls, id):
        client = Client.query.get(id)
        return add_ok({"client": _dict_or_empty(client)})

    @classmethod
    def patch_client_by_project_admin(cls, id, req):
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
        try:
            db.session.add(client)
            db.session.commit()
        except Exception as e:
            log.error(f"Error while patching client: {e}")
            return None
        return add_ok({"client": _dict_or_empty(client)})

    @classmethod
    def patch_client_by_creator(cls, id, req):
        client = Client.query.get(id)
        _ = req.pop("approval_state", None)
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
        try:
            db.session.add(client)
            db.session.commit()
        except Exception as e:
            log.error(f"Error while patching client: {e}")
            return None
        return add_ok({"client": _dict_or_empty(client)})

    @classmethod
    def delete_client(cls, id):
        client = Client.query.get(id)
        db.session.delete(client)
        db.session.commit()
        return add_ok({})

    @classmethod
    def get_client_blob(cls, key, id):
        fileobj, filename = gen_client(key, id)
        inc_dl(Client, id)
        return fileobj, filename

    @classmethod
    def create_user(cls, req):
        name = req.get("name", "")
        email = req.get("email")
        password = req.get("password", "")
        password_hash = generate_password_hash(password)
        organization = req.get("organization", "")
        role_name = req.get("role", "")
        description = req.get("description", "")
        approval_state = req.get("approval_state", 0)
        org = get_or_create(db.session, Organization, name=organization)
        role = get_or_create(db.session, Role, name=role_name)
        try:
            user = User(
                email=email,
                name=name,
                password_hash=password_hash,
                description=description,
                approval_state=approval_state,
            )
            user.organization_id = org.id
            user.role_id = role.id
            db.session.add(user)
            db.session.commit()
        except Exception as e:
            log.error(f"Error while creating user: {e}")
            return None
        return add_ok({"user": _dict_or_empty(user)})

    @classmethod
    def verify_user(cls, email, password):
        user = User.query.filter_by(email=email).first()
        if user is not None and check_password_hash(user.password_hash, password):
            return user
        else:
            return None

    @classmethod
    def get_users(cls, org_name=None):
        if org_name is None:
            all_users = User.query.all()
        else:
            org = Organization.query.filter_by(name=org_name).first()
            if org:
                all_users = org.users
            else:
                all_users = {}
        return add_ok({"user_list": [_dict_or_empty(user) for user in all_users]})

    @classmethod
    def _get_email_by_id(cls, id):
        user = User.query.get(id)
        return user.email if user else None

    @classmethod
    def get_user(cls, id):
        user = User.query.get(id)
        return add_ok({"user": _dict_or_empty(user)})

    @classmethod
    def patch_user_by_project_admin(cls, id, req):
        user = User.query.get(id)
        org_name = req.pop("organization", None)
        if org_name is not None:
            org = get_or_create(db.session, Organization, name=org_name)
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

    @classmethod
    def patch_user_by_creator(cls, id, req):
        user = User.query.get(id)
        _ = req.pop("approval_state", None)
        role = req.pop("role", None)
        if role is not None and user.role.name == "":
            role = get_or_create(db.session, Role, name=role)
            user.role_id = role.id
        organization = req.pop("organization", None)
        if organization is not None and user.organization.name == "":
            org = get_or_create(db.session, Organization, name=organization)
            user.organization_id = org.id
        password = req.pop("password", None)
        if password is not None:
            password_hash = generate_password_hash(password)
            user.password_hash = password_hash
        for k, v in req.items():
            setattr(user, k, v)
        db.session.add(user)
        db.session.commit()
        return add_ok({"user": _dict_or_empty(user)})

    @classmethod
    def delete_user(cls, id):
        clients = Client.query.filter_by(creator_id=id).all()
        for client in clients:
            db.session.delete(client)
        user = User.query.get(id)
        db.session.delete(user)
        db.session.commit()

        return add_ok({})

    @classmethod
    def get_user_blob(cls, key, id):
        fileobj, filename = gen_user(key, id)
        inc_dl(User, id)
        return fileobj, filename

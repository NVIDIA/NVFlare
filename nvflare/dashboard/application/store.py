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

from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

from .cert import Entity, make_root_cert
from .models import Client, Organization, Project, Role, User, db
from .validation import CLIENT_NAMES, SERVER_NAMES, USER_NAMES, DashboardError, validate_names

log = logging.getLogger(__name__)

_PROJECT_WRITABLE = {
    "title",
    "description",
    "app_location",
    "starting_date",
    "end_date",
    "server1",
    "frozen",
    "public",
    "cc_mode",
}

_REQUESTABLE_ROLES = {"", "lead", "member", "org_admin"}


def is_project_admin(user):
    return user is not None and user.approval_state >= 100 and user.role.name == "project_admin"


def check_role(id, claims, requester):
    is_creator = requester == Store._get_email_by_id(id)
    # JWT roles/approval can be stale after an administrator edits an account.
    user = User.query.filter_by(email=requester).first()
    return is_creator and user is not None and user.approval_state >= 0, is_project_admin(user)


def _dict_or_empty(item):
    return item.asdict() if item else {}


def get_or_create(session, model, **kwargs):
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance
    else:
        instance = model(**kwargs)
        session.add(instance)
        # The caller commits the complete operation, including any approval checks.
        session.flush()
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
    def seed_user(cls, email, pwd, org):
        seed_user = {
            "name": "super_name",
            "email": email,
            "password": pwd,
            "organization": org,
            "role": "project_admin",
            "approval_state": 200,
        }
        cls.create_user(seed_user, seeding=True)
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
        validate_names(req, SERVER_NAMES, allow_empty=("server1",))
        if req.get("frozen"):
            validate_names({"server1": req.get("server1", project.server1)}, SERVER_NAMES)
        req.pop("id", None)

        short_name = req.pop("short_name", "")
        if short_name:
            if len(short_name) > 16:
                short_name = short_name[:16]
            project.short_name = short_name

        proj_props = req.pop("project_props", None)
        if proj_props:
            project.project_props = json.dumps(proj_props)

        server_props = req.pop("server_props", None)
        if server_props:
            project.server_props = json.dumps(server_props)

        for k, v in req.items():
            if k in _PROJECT_WRITABLE:
                setattr(project, k, v)

        db.session.add(project)
        if project.frozen:
            # Publish the frozen state and its CA material in the same commit.
            cls.build_project(project)
        else:
            db.session.commit()

        project_dict = _dict_or_empty(project)
        project_dict = cls._add_registered_info(project_dict)
        return add_ok({"project": project_dict})

    @classmethod
    def get_project(cls):
        project_dict = _dict_or_empty(Project.query.first())
        project_dict = cls._add_registered_info(project_dict)
        return add_ok({"project": project_dict})

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
        return user is not None and user.approval_state >= 100

    @classmethod
    def get_requester(cls, email):
        user = User.query.filter_by(email=email).first()
        if user is None or user.approval_state < 0:
            raise DashboardError("This account is unavailable or has been denied.")
        return user

    @classmethod
    def require_site_owner(cls, user, client=None):
        """Pending applicants may manage only their own pending site registrations."""
        if is_project_admin(user):
            return
        if user.approval_state < 0 or user.role.name != "org_admin" or not user.organization.name:
            raise DashboardError("Request the Org Admin role and an organization before registering client sites.")
        if client is not None:
            if client.creator_id != user.id or client.organization_id != user.organization_id:
                raise DashboardError("You can access only your own client sites in your organization.")
            if user.approval_state < 100 and client.approval_state != 0:
                raise DashboardError("Pending applicants can manage only their pending client sites.")

    @classmethod
    def create_client(cls, req, creator):
        validate_names(req, CLIENT_NAMES, required=CLIENT_NAMES)
        user = cls.get_requester(creator)
        cls.require_site_owner(user)
        if not is_project_admin(user) and req["organization"] != user.organization.name:
            raise DashboardError("Client sites must use your registered organization.")
        creator_id = user.id
        name = req.get("name")
        organization = req.get("organization", "")
        description = req.get("description", "")
        org = get_or_create(db.session, Organization, name=organization)
        client = Client(name=name, description=description, creator_id=creator_id)
        client.organization_id = org.id

        capacity = req.get("capacity")
        if capacity:
            client.capacity = json.dumps(capacity)

        props = req.get("props")
        if props:
            client.props = json.dumps(props)
        else:
            client.props = ""
        try:
            db.session.add(client)
            db.session.commit()
        except Exception as e:
            log.error(f"Error while creating client: {e}")
            return None
        return add_ok({"client": _dict_or_empty(client)})

    @classmethod
    def get_clients(cls, org=None, creator_id=None, pending_only=False):
        if creator_id is not None:
            query = Client.query.filter_by(creator_id=creator_id)
            if org is not None:
                query = query.filter(Client.organization.has(name=org))
            if pending_only:
                query = query.filter_by(approval_state=0)
            all_clients = query.all()
        elif org is None:
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
        validate_names(req, CLIENT_NAMES)
        client = Client.query.get(id)

        organization = req.pop("organization", None)
        if organization is not None:
            org = get_or_create(db.session, Organization, name=organization)
            client.organization_id = org.id

        capacity = req.pop("capacity", None)
        if capacity:
            client.capacity = json.dumps(capacity)

        props = req.pop("props", None)
        if props:
            client.props = json.dumps(props)

        for k, v in req.items():
            if k in {"name", "description", "approval_state"}:
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
        validate_names(req, {"name": "client"})
        client = Client.query.get(id)
        _ = req.pop("approval_state", None)
        _ = req.pop("organization", None)

        values = {k: v for k, v in req.items() if k in {"name", "description"}}
        capacity = req.pop("capacity", None)
        if capacity:
            values["capacity"] = json.dumps(capacity)

        props = req.pop("props", None)
        if props:
            values["props"] = json.dumps(props)
        if not values:
            return add_ok({"client": _dict_or_empty(client)})
        try:
            result = db.session.execute(
                db.update(Client)
                .where(
                    Client.id == client.id,
                    Client.creator_id == client.creator_id,
                    Client.organization_id == client.organization_id,
                    Client.approval_state == client.approval_state,
                )
                .values(**values)
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                raise DashboardError(
                    "The client site changed. Reload its current approval and details before editing.", status_code=409
                )
            db.session.commit()
        except IntegrityError as error:
            db.session.rollback()
            message = "This site name already exists. Please choose another site name."
            raise DashboardError(message, status_code=409, errors={"name": message}) from error
        db.session.refresh(client)
        return add_ok({"client": _dict_or_empty(client)})

    @classmethod
    def delete_client(cls, id, creator_id=None):
        client = Client.query.get(id)
        if creator_id is None:
            db.session.delete(client)
        else:
            result = db.session.execute(
                db.delete(Client)
                .where(
                    Client.id == client.id,
                    Client.creator_id == creator_id,
                    Client.organization_id == client.organization_id,
                    Client.approval_state == client.approval_state,
                    Client.approval_state < 100,
                )
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                raise DashboardError(
                    "The client site changed. Reload its current approval before deleting.", status_code=409
                )
        db.session.commit()
        return add_ok({})

    @classmethod
    def create_user(cls, req, seeding=False, created_by_project_admin=False):
        name = req.get("name", "")
        email = req.get("email")
        password = req.get("password", "")
        password_hash = generate_password_hash(password)
        organization = req.get("organization", "")
        role_name = req.get("role", "")
        description = req.get("description", "")
        if not isinstance(role_name, str):
            raise DashboardError(
                "Role must be a string.", status_code=400, errors={"role": "Select Member, Lead, or Org Admin."}
            )
        if seeding:
            approval_state = 200
        else:
            if not created_by_project_admin and role_name not in _REQUESTABLE_ROLES:
                raise DashboardError(
                    "That role cannot be requested during registration.",
                    errors={"role": "Select Member, Lead, or Org Admin."},
                )
            approval_state = 0
        validate_names(
            req,
            USER_NAMES,
            required=("email", "organization", "role") if seeding else ("email",),
            allow_empty=() if seeding else ("organization", "role"),
        )
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
            if user.approval_state < 0:
                return None
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
    def _commit_user_patch(cls, user, values):
        # Compare-and-swap works on SQLite too: role/organization changes cannot
        # race an approval and retain approval of a different registration.
        result = db.session.execute(
            db.update(User)
            .where(
                User.id == user.id,
                User.role_id == user.role_id,
                User.organization_id == user.organization_id,
                User.approval_state == user.approval_state,
            )
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            raise DashboardError("The registration changed. Reload it and review it again.", status_code=409)
        db.session.commit()
        db.session.refresh(user)
        return add_ok({"user": _dict_or_empty(user)})

    @classmethod
    def patch_user_by_project_admin(cls, id, req):
        validate_names(req, {"organization": "org", "role": "simple_name"})
        user = db.session.get(User, id)
        if user is None:
            raise DashboardError("User not found.", status_code=404)
        values = {k: v for k, v in req.items() if k in {"name", "description", "approval_state"}}
        approval = values.get("approval_state", user.approval_state)
        if not isinstance(approval, int) or isinstance(approval, bool):
            raise DashboardError("Approval state must be an integer.", status_code=400)
        if "approval_state" in req and approval >= 100:
            validate_names(
                {
                    "email": user.email,
                    "role": req.get("role", user.role.name),
                    "organization": req.get("organization", user.organization.name),
                },
                USER_NAMES,
                required=USER_NAMES,
            )
            if user.approval_state < 100 and (
                req.get("reviewed_role") != user.role.name or req.get("reviewed_organization") != user.organization.name
            ):
                raise DashboardError(
                    "Review the current role and organization before approving this user.", status_code=409
                )
        if "organization" in req:
            org = get_or_create(db.session, Organization, name=req["organization"])
            if org.id != user.organization_id and Client.query.filter_by(creator_id=user.id).count():
                raise DashboardError(
                    "Update or remove this user's client sites before changing their organization.", status_code=409
                )
            values["organization_id"] = org.id
        if "role" in req:
            values["role_id"] = get_or_create(db.session, Role, name=req["role"]).id
        if "password" in req:
            values["password_hash"] = generate_password_hash(req["password"])
        return cls._commit_user_patch(user, values)

    @classmethod
    def patch_user_by_creator(cls, id, req):
        user = User.query.get(id)
        if user.approval_state < 0:
            raise DashboardError("This registration has been denied. Contact your project administrator.")
        role = req.get("role")
        if "role" in req and (not isinstance(role, str) or role not in _REQUESTABLE_ROLES):
            raise DashboardError(
                "That role cannot be requested during registration.",
                errors={"role": "Select Member, Lead, or Org Admin."},
            )
        validate_names(req, {"organization": "org"})
        values = {k: v for k, v in req.items() if k in {"name", "description"}}
        for field, current in (("role", user.role.name), ("organization", user.organization.name)):
            if field in req and req[field] != current:
                if user.approval_state >= 100 or current:
                    raise DashboardError(
                        "Only the project administrator can change an existing or approved role or organization."
                    )
                if field == "role":
                    values["role_id"] = get_or_create(db.session, Role, name=role).id
                else:
                    values["organization_id"] = get_or_create(db.session, Organization, name=req[field]).id
        if "password" in req:
            values["password_hash"] = generate_password_hash(req["password"])
        if not values:
            return add_ok({"user": _dict_or_empty(user)})
        return cls._commit_user_patch(user, values)

    @classmethod
    def delete_user(cls, id):
        clients = Client.query.filter_by(creator_id=id).all()
        for client in clients:
            db.session.delete(client)
        user = User.query.get(id)
        db.session.delete(user)
        db.session.commit()

        return add_ok({})

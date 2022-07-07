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

import json
import os

from nvflare.lighter.spec import Builder, Project


class AuthPolicyBuilder(Builder):
    def __init__(self, orgs, roles, groups, disabled):
        """Build authorization.json.

        Creates and writes authorization.json to the server's startup directory with the authorization policy
        defining the groups each org is in and the admin client roles which controls the allowed rights. The
        participant information from project.yml is included in authorization.json with what orgs, groups, and roles
        are associated with each participant. This builder also checks for errors if the arguments are specified
        incorrectly.

        Args:
            orgs: authorization configuration for orgs (it may be helpful to build this section with the UI)
            roles: authorization configuration for roles (it may be helpful to build this section with the UI)
            groups: authorization configuration for groups (it may be helpful to build this section with the UI)
            disabled: if true, all users are super with all privileges
        """
        self.orgs = orgs
        self.roles = roles
        self.groups = groups
        self.disabled = disabled

    def build(self, project: Project, ctx: dict):
        authz = {"version": "1.0"}
        authz["roles"] = self.roles
        authz["groups"] = self.groups
        users = dict()
        for admin in project.get_participants_by_type("admin", first_only=False):
            if admin.org not in self.orgs:
                raise ValueError(f"Admin {admin.name}'s org {admin.org} not defined in AuthPolicy")
            if self.disabled:
                users[admin.name] = {"org": admin.org, "roles": ["super"]}
            else:
                for role in admin.props.get("roles", {}):
                    if role not in self.roles:
                        raise ValueError(f"Admin {admin.name}'s role {role} not defined in AuthPolicy")
                users[admin.name] = {"org": admin.org, "roles": admin.props.get("roles")}
        authz["users"] = users
        authz["orgs"] = self.orgs
        servers = project.get_participants_by_type("server", first_only=False)
        for server in servers:
            if server.org not in self.orgs:
                raise ValueError(f"Server {server.name}'s org {server.org} not defined in AuthPolicy")
            sites = {"server": server.org}
            for client in project.get_participants_by_type("client", first_only=False):
                if client.org not in self.orgs:
                    raise ValueError(f"client {client.name}'s org {client.org} not defined in AuthPolicy")
                sites[client.name] = client.org
            authz["sites"] = sites
            authz.update(ctx.get("authz_def"))
            dest_dir = self.get_kit_dir(server, ctx)
            with open(os.path.join(dest_dir, "authorization.json"), "wt") as f:
                f.write(json.dumps(authz))

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import shutil
import traceback
from abc import ABC
from typing import List

import yaml

from nvflare.apis.utils.format_check import name_check
from nvflare.lighter import utils

from .constants import AdminRole, CtxKey, ParticipantType, PropKey, ProvisionMode, WorkDir


def _check_host_name(scope: str, prop_key: str, value):
    err, reason = name_check(value, "host_name")
    if err:
        raise ValueError(f"bad value for {prop_key} '{value}' in {scope}: {reason}")


def _check_host_names(scope: str, prop_key: str, value):
    if isinstance(value, str):
        _check_host_name(scope, prop_key, value)
    elif isinstance(value, list):
        for v in value:
            _check_host_name(scope, prop_key, v)


def _check_admin_role(scope: str, prop_key: str, value):
    valid_roles = [AdminRole.PROJECT_ADMIN, AdminRole.ORG_ADMIN, AdminRole.LEAD, AdminRole.MEMBER]
    if value not in valid_roles:
        raise ValueError(f"bad value for {prop_key} '{value}' in {scope}: must be one of {valid_roles}")


# validator functions for common properties
# Validator function must follow this signature:
# func(scope: str, prop_key: str, value)
_PROP_VALIDATORS = {
    PropKey.HOST_NAMES: _check_host_names,
    PropKey.CONNECT_TO: _check_host_name,
    PropKey.LISTENING_HOST: _check_host_name,
    PropKey.DEFAULT_HOST: _check_host_name,
    PropKey.ROLE: _check_admin_role,
}


class Entity:
    def __init__(self, scope: str, name: str, props: dict, parent=None):
        if not props:
            props = {}

        for k, v in props.items():
            validator = _PROP_VALIDATORS.get(k)
            if validator is not None:
                validator(scope, k, v)
        self.name = name
        self.props = props
        self.parent = parent

    def get_prop(self, key: str, default=None):
        return self.props.get(key, default)

    def get_prop_fb(self, key: str, fb_key=None, default=None):
        """Get property value with fallback.
        If I have the property, then return it.
        If not, I return the fallback property of my parent. If I don't have parent, return default.

        Args:
            key: key of the property
            fb_key: key of the fallback property.
            default: value to return if no one has the property

        Returns: property value

        """
        value = self.get_prop(key)
        if value:
            return value
        elif not self.parent:
            return default
        else:
            # get the value from the parent
            if not fb_key:
                fb_key = key
            return self.parent.get_prop(fb_key, default)


class Participant(Entity):
    def __init__(self, type: str, name: str, org: str, props: dict = None, project: Entity = None):
        """Class to represent a participant.

        Each participant communicates to other participant.  Therefore, each participant has its
        own name, type, organization it belongs to, rules and other information.

        Args:
            type (str): server, client, admin or other string that builders can handle
            name (str): system-wide unique name
            org (str): system-wide unique organization
            props (dict): properties
            project: the project that the participant belongs to

        Raises:
            ValueError: if name or org is not compliant with characters or format specification.
        """
        Entity.__init__(self, f"{type}::{name}", name, props, parent=project)

        err, reason = name_check(name, type)
        if err:
            raise ValueError(reason)

        err, reason = name_check(org, "org")
        if err:
            raise ValueError(reason)

        self.type = type
        self.org = org
        self.subject = name

    def get_default_host(self) -> str:
        """Get the default host name for accessing this participant (server).
        If the "default_host" attribute is explicitly specified, then it's the default host.
        If the "default_host" attribute is not explicitly specified, then use the "name" attribute.

        Returns: a host name

        """
        h = self.get_prop(PropKey.DEFAULT_HOST)
        if h:
            return h
        else:
            return self.name


class Project(Entity):
    def __init__(
        self,
        name: str,
        description: str,
        participants=None,
        props: dict = None,
        serialized_root_cert=None,
        serialized_root_private_key=None,
    ):
        """A container class to hold information about this FL project.

        This class only holds information.  It does not drive the workflow.

        Args:
            name (str): the project name
            description (str): brief description on this name
            participants: if provided, list of participants of the project
            props: properties of the project
            serialized_root_cert: if provided, the root cert to be used for the project
            serialized_root_private_key: if provided, the root private key for signing certs of sites and admins

        Raises:
            ValueError: when participant criteria is violated
        """
        Entity.__init__(self, "project", name, props)

        if serialized_root_cert:
            if not serialized_root_private_key:
                raise ValueError("missing serialized_root_private_key while serialized_root_cert is provided")

        self.description = description
        self.serialized_root_cert = serialized_root_cert
        self.serialized_root_private_key = serialized_root_private_key
        self.server = None
        self.overseer = None
        self.clients = []
        self.admins = []
        self.all_names = {}

        if participants:
            if not isinstance(participants, list):
                raise ValueError(f"participants must be a list of Participant but got {type(participants)}")

            for p in participants:
                if not isinstance(p, Participant):
                    raise ValueError(f"bad item in participants: must be Participant but got {type(p)}")

                if p.type == ParticipantType.SERVER:
                    self.set_server(p.name, p.org, p.props)
                elif p.type == ParticipantType.ADMIN:
                    self.add_admin(p.name, p.org, p.props)
                elif p.type == ParticipantType.CLIENT:
                    self.add_client(p.name, p.org, p.props)
                elif p.type == ParticipantType.OVERSEER:
                    self.set_overseer(p.name, p.org, p.props)
                else:
                    raise ValueError(f"invalid value for ParticipantType: {p.type}")

    def _check_unique_name(self, name: str):
        if name in self.all_names:
            raise ValueError(f"the project {self.name} already has a participant with the name '{name}'")

    def set_server(self, name: str, org: str, props: dict):
        if self.server:
            raise ValueError(f"project {self.name} already has a server defined")
        self._check_unique_name(name)
        self.server = Participant(ParticipantType.SERVER, name, org, props, self)
        self.all_names[name] = True

    def get_server(self):
        """Get the server definition. Only one server is supported!

        Returns: server participant

        """
        return self.server

    def set_overseer(self, name: str, org: str, props: dict):
        if self.overseer:
            raise ValueError(f"project {self.name} already has an overseer defined")
        self._check_unique_name(name)
        self.overseer = Participant(ParticipantType.OVERSEER, name, org, props, self)
        self.all_names[name] = True

    def get_overseer(self):
        """Get the overseer definition. Only one overseer is supported!

        Returns: overseer participant

        """
        return self.overseer

    def add_client(self, name: str, org: str, props: dict):
        self._check_unique_name(name)
        self.clients.append(Participant(ParticipantType.CLIENT, name, org, props, self))
        self.all_names[name] = True

    def get_clients(self):
        return self.clients

    def add_admin(self, name: str, org: str, props: dict):
        self._check_unique_name(name)
        admin = Participant(ParticipantType.ADMIN, name, org, props, self)
        role = admin.get_prop(PropKey.ROLE)
        if not role:
            raise ValueError(f"missing role in admin '{name}'")
        self.admins.append(admin)
        self.all_names[name] = True

    def get_admins(self):
        return self.admins

    def get_all_participants(self):
        result = []
        if self.server:
            result.append(self.server)

        if self.overseer:
            result.append(self.overseer)

        result.extend(self.clients)
        result.extend(self.admins)
        return result


class ProvisionContext(dict):
    def __init__(self, workspace_root_dir: str, project: Project):
        super().__init__()
        self[CtxKey.WORKSPACE] = workspace_root_dir

        wip_dir = os.path.join(workspace_root_dir, "wip")
        state_dir = os.path.join(workspace_root_dir, "state")
        resources_dir = os.path.join(workspace_root_dir, "resources")
        self.update({CtxKey.WIP: wip_dir, CtxKey.STATE: state_dir, CtxKey.RESOURCES: resources_dir})
        dirs = [workspace_root_dir, resources_dir, wip_dir, state_dir]
        utils.make_dirs(dirs)

        # set commonly used data into ctx
        self[CtxKey.PROJECT] = project

        server = project.get_server()
        admin_port = server.get_prop(PropKey.ADMIN_PORT, 8003)
        self[CtxKey.ADMIN_PORT] = admin_port
        fed_learn_port = server.get_prop(PropKey.FED_LEARN_PORT, 8002)
        self[CtxKey.FED_LEARN_PORT] = fed_learn_port
        self[CtxKey.SERVER_NAME] = server.name

    def get_project(self):
        return self.get(CtxKey.PROJECT)

    def set_template(self, template: dict):
        self[CtxKey.TEMPLATE] = template

    def get_template(self):
        return self.get(CtxKey.TEMPLATE)

    def get_template_section(self, section_key: str):
        template = self.get_template()
        if not template:
            raise RuntimeError("template is not available")

        section = template.get(section_key)
        if not section:
            raise RuntimeError(f"missing section {section} in template")

        return section

    def set_provision_mode(self, mode: str):
        valid_modes = [ProvisionMode.POC, ProvisionMode.NORMAL]
        if mode not in valid_modes:
            raise ValueError(f"invalid provision mode {mode}: must be one of {valid_modes}")
        self[CtxKey.PROVISION_MODE] = mode

    def get_provision_mode(self):
        return self.get(CtxKey.PROVISION_MODE)

    def get_wip_dir(self):
        return self.get(CtxKey.WIP)

    def get_ws_dir(self, entity: Entity):
        return os.path.join(self.get_wip_dir(), entity.name)

    def get_kit_dir(self, entity: Entity):
        return os.path.join(self.get_ws_dir(entity), "startup")

    def get_transfer_dir(self, entity: Entity):
        return os.path.join(self.get_ws_dir(entity), "transfer")

    def get_local_dir(self, entity: Entity):
        return os.path.join(self.get_ws_dir(entity), "local")

    def get_state_dir(self):
        return self.get(CtxKey.STATE)

    def get_resources_dir(self):
        return self.get(CtxKey.RESOURCES)

    def get_workspace(self):
        return self.get(CtxKey.WORKSPACE)

    def yaml_load_template_section(self, section_key: str):
        return yaml.safe_load(self.get_template_section(section_key))

    def json_load_template_section(self, section_key: str):
        return json.loads(self.get_template_section(section_key))


class Builder(ABC):
    def initialize(self, project: Project, ctx: ProvisionContext):
        pass

    def build(self, project: Project, ctx: ProvisionContext):
        pass

    def finalize(self, project: Project, ctx: ProvisionContext):
        pass

    @staticmethod
    def build_from_template(
        ctx: ProvisionContext, dest_dir: str, temp_section: str, file_name, replacement=None, mode="t", exe=False
    ):
        section = ctx.get_template_section(temp_section)
        if replacement:
            section = utils.sh_replace(section, replacement)
        utils.write(os.path.join(dest_dir, file_name), section, mode, exe=exe)


class Provisioner(object):
    def __init__(self, root_dir: str, builders: List[Builder]):
        """Workflow class that drive the provision process.

        Provisioner's tasks:

            - Maintain the provision workspace folder structure;
            - Invoke Builders to generate the content of each startup kit

        ROOT_WORKSPACE Folder Structure::

            root_workspace_dir_name: this is the root of the workspace
                project_dir_name: the root dir of the project, could be named after the project
                    resources: stores resource files (templates, configs, etc.) of the Provisioner and Builders
                    prod: stores the current set of startup kits (production)
                        participate_dir: stores content files generated by builders
                    wip: stores the set of startup kits to be created (WIP)
                        participate_dir: stores content files generated by builders
                    state: stores the persistent state of the Builders

        Args:
            root_dir (str): the directory path to hold all generated or intermediate folders
            builders (List[Builder]): all builders that will be called to build the content
        """
        self.root_dir = root_dir
        self.builders = builders
        self.template = {}

    def add_template(self, template: dict):
        if not isinstance(template, dict):
            raise ValueError(f"template must be a dict but got {type(template)}")
        self.template.update(template)

    def provision(self, project: Project, mode=None):
        server = project.get_server()
        if not server:
            raise RuntimeError("missing server from the project")

        workspace_root_dir = os.path.join(self.root_dir, project.name)
        ctx = ProvisionContext(workspace_root_dir, project)
        if self.template:
            ctx.set_template(self.template)

        if not mode:
            mode = ProvisionMode.NORMAL
        ctx.set_provision_mode(mode)

        try:
            for b in self.builders:
                b.initialize(project, ctx)

            # call builders!
            for b in self.builders:
                b.build(project, ctx)

            for b in self.builders[::-1]:
                b.finalize(project, ctx)

        except Exception:
            prod_dir = ctx.get(WorkDir.CURRENT_PROD_DIR)
            if prod_dir:
                shutil.rmtree(prod_dir)
            print("Exception raised during provision.  Incomplete prod_n folder removed.")
            traceback.print_exc()
        finally:
            wip_dir = ctx.get(WorkDir.WIP)
            if wip_dir:
                shutil.rmtree(wip_dir)
        return ctx

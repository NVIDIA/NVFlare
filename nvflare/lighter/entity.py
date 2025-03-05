# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any, List, Optional, Union

from nvflare.apis.utils.format_check import name_check

from .constants import DEFINED_PARTICIPANT_TYPES, DEFINED_ROLES, ConnSecurity, ParticipantType, PropKey


class ListeningHost:
    def __init__(self, scheme, host_names, default_host, port, conn_sec):
        self.scheme = scheme
        self.host_names = host_names
        self.default_host = default_host
        self.port = port
        self.conn_sec = conn_sec

    def __str__(self):
        scheme, host_names, default_host, port, conn_sec = (
            self.scheme,
            self.host_names,
            self.default_host,
            self.port,
            self.conn_sec,
        )
        return f"ListeningHost[{scheme=} {host_names=} {default_host=} {port=} {conn_sec=}]"


class ConnectTo:
    def __init__(self, name, host, port, conn_sec):
        self.name = name
        self.host = host
        self.port = port
        self.conn_sec = conn_sec

    def __str__(self):
        name, host, port, conn_sec = self.name, self.host, self.port, self.conn_sec
        return f"ConnectTo[{name=} {host=} {port=} {conn_sec=}]"


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
    if not isinstance(value, str):
        raise ValueError(f"bad value for {prop_key} '{value}' in {scope}: must be str but got {type(value)}")

    if not value:
        raise ValueError(f"empty value for {prop_key} '{value}' in {scope}")


def parse_connect_to(value, scope=None, prop_key=None) -> ConnectTo:
    """Parse the "connect_to" property.

    Args:
        value: value to be parsed. It is either a str or a dict.
        scope: scope of the property
        prop_key: key of the property

    Returns: a ConnectTo object

    """
    if isinstance(value, str):
        # old format - for server only
        return ConnectTo(None, value, None, None)
    elif isinstance(value, dict):
        name = value.get(PropKey.NAME)
        host = value.get(PropKey.HOST)
        port = value.get(PropKey.PORT)
        conn_sec = value.get(PropKey.CONN_SECURITY)
        return ConnectTo(name, host, port, conn_sec)
    else:
        raise ValueError(
            f"bad value for {prop_key} '{value}' in {scope}: invalid type {type(value)}; must be str or dict"
        )


def _check_connect_to(scope: str, prop_key: str, value):
    ct = parse_connect_to(value, scope, prop_key)
    if ct.host:
        err, reason = name_check(ct.host, "host_name")
        if err:
            raise ValueError(f"bad value for {prop_key} '{value}' in {scope}: {reason}")

    if ct.port is not None:
        if not isinstance(ct.port, int):
            raise ValueError(f"bad value for {prop_key} '{value}' in {scope}: port {ct.port} must be int")

        if ct.port < 0:
            raise ValueError(f"bad value for {prop_key} '{value}' in {scope}: invalid port {ct.port}")


def _check_conn_security(scope: str, prop_key: str, value):
    if not isinstance(value, str):
        raise ValueError(f"bad value for {prop_key} '{value}' in {scope}: must be a str but got {type(value)}")

    valid_conn_secs = [ConnSecurity.CLEAR, ConnSecurity.MTLS, ConnSecurity.TLS]
    if value.lower() not in valid_conn_secs:
        raise ValueError(f"bad value for {prop_key} '{value}' in {scope}: must be one of {valid_conn_secs}")


def parse_listening_host(value, scope=None, prop_key=None) -> ListeningHost:
    """Parse the "listening_host" property. It must be either str or a dict

    Args:
        value: value to be parsed
        scope: scope of the prop
        prop_key: key of the property

    Returns: a ListeningHost object
    """
    if isinstance(value, str):
        # old format - for server only
        return ListeningHost(None, None, value, None, None)
    elif isinstance(value, dict):
        scheme = value.get(PropKey.SCHEME)
        host_names = value.get(PropKey.HOST_NAMES)
        default_host = value.get(PropKey.DEFAULT_HOST)
        port = value.get(PropKey.PORT)
        conn_sec = value.get(PropKey.CONN_SECURITY)
        return ListeningHost(scheme, host_names, default_host, port, conn_sec)
    else:
        raise ValueError(
            f"bad value for {prop_key} '{value}' in {scope}: invalid type {type(value)}; must be str or dict"
        )


def _check_listening_host(scope: str, prop_key: str, value):
    h = parse_listening_host(value, scope, prop_key)
    if h.host_names:
        _check_host_names(scope, prop_key, h.host_names)

    if h.default_host:
        _check_host_name(scope, prop_key, h.default_host)

    if h.port is not None:
        if not isinstance(h.port, int):
            raise ValueError(f"bad value for {prop_key} '{value}' in {scope}: port {h.port} must be int")

        if h.port < 0:
            raise ValueError(f"bad value for {prop_key} '{value}' in {scope}: invalid port {h.port}")


# validator functions for common properties
# Validator function must follow this signature:
# func(scope: str, prop_key: str, value)
_PROP_VALIDATORS = {
    PropKey.HOST_NAMES: _check_host_names,
    PropKey.CONNECT_TO: _check_connect_to,
    PropKey.LISTENING_HOST: _check_listening_host,
    PropKey.DEFAULT_HOST: _check_host_name,
    PropKey.ROLE: _check_admin_role,
    PropKey.CONN_SECURITY: _check_conn_security,
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

    def set_prop(self, key: str, value: Any):
        self.props[key] = value

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
            type (str): server, client, admin, relay or other string that builders can handle
            name (str): system-wide unique name
            org (str): system-wide unique organization
            props (dict): properties
            project: the project that the participant belongs to

        Raises:
            ValueError: if name or org is not compliant with characters or format specification.
        """
        Entity.__init__(self, f"{type}::{name}", name, props, parent=project)

        if type in DEFINED_PARTICIPANT_TYPES:
            err, reason = name_check(name, type)
            if err:
                raise ValueError(reason)
        else:
            print(f"Warning: participant type '{type}' of {name} is not a defined type {DEFINED_PARTICIPANT_TYPES}")

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

    def get_listening_host(self) -> Optional[ListeningHost]:
        h = self.get_prop(PropKey.LISTENING_HOST)
        if not h:
            return None

        lh = parse_listening_host(h)
        if not lh.scheme:
            lh.scheme = "tcp"

        if not lh.port:
            lh.port = 0  # any port

        if not lh.conn_sec:
            lh.conn_sec = ConnSecurity.CLEAR

        if not lh.default_host:
            if self.type == ParticipantType.SERVER:
                lh.default_host = self.get_default_host()
            else:
                lh.default_host = "localhost"

        return lh

    def get_connect_to(self) -> Optional[ConnectTo]:
        h = self.get_prop(PropKey.CONNECT_TO)
        if not h:
            return None
        else:
            return parse_connect_to(h)


def participant_from_dict(participant_def: dict, project=None) -> Participant:
    name = participant_def.pop(PropKey.NAME, None)
    if not name:
        raise ValueError("missing participant name")

    t = participant_def.pop(PropKey.TYPE, None)
    if not t:
        raise ValueError("missing participant type")

    org = participant_def.pop(PropKey.ORG, None)
    if not org:
        raise ValueError("missing participant org")

    return Participant(type=t, name=name, org=org, props=participant_def, project=project)


class Project(Entity):
    def __init__(
        self,
        name: str,
        description: str,
        participants=None,
        props: dict = None,
        serialized_root_cert=None,
        root_private_key=None,
    ):
        """A container class to hold information about this FL project.

        This class only holds information.  It does not drive the workflow.

        Args:
            name (str): the project name
            description (str): brief description on this name
            participants: if provided, list of participants of the project
            props: properties of the project
            serialized_root_cert: if provided, the root cert to be used for the project
            root_private_key: if provided, the root private key for signing certs of sites and admins

        Raises:
            ValueError: when participant criteria is violated
        """
        Entity.__init__(self, "project", name, props)

        if serialized_root_cert:
            if not root_private_key:
                raise ValueError("missing root_private_key while serialized_root_cert is provided")

        self.description = description
        self.serialized_root_cert = serialized_root_cert
        self.root_private_key = root_private_key
        self.server = None
        self.overseer = None
        self.participants_by_types = {}  # participant type => list of participants
        self.all_names = {}  # name => participant

        if participants:
            if not isinstance(participants, list):
                raise ValueError(f"participants must be a list of Participant but got {type(participants)}")

            for p in participants:
                if not isinstance(p, Participant):
                    raise ValueError(f"bad item in participants: must be Participant but got {type(p)}")
                self.add_participant(p)

    def set_server(self, name: str, org: str, props: dict):
        return self.add_participant(Participant(ParticipantType.SERVER, name, org, props))

    def get_server(self):
        """Get the server definition. Only one server is supported!

        Returns: server participant

        """
        return self.server

    def set_overseer(self, name: str, org: str, props: dict):
        return self.add_participant(Participant(ParticipantType.OVERSEER, name, org, props))

    def get_overseer(self):
        """Get the overseer definition. Only one overseer is supported!

        Returns: overseer participant

        """
        return self.overseer

    def add_participant(self, participant: Participant):
        if participant.name in self.all_names:
            raise ValueError(f"the project {self.name} already has a participant with the name '{participant.name}'")

        participant.parent = self
        if participant.type == ParticipantType.SERVER:
            if self.server:
                raise ValueError(f"cannot add participant {participant.name} as server - server already exists")
            self.server = participant
        elif participant.type == ParticipantType.OVERSEER:
            if self.overseer:
                raise ValueError(f"cannot add participant {participant.name} as overseer - overseer already exists")
            self.overseer = participant
        elif participant.type == ParticipantType.ADMIN:
            role = participant.get_prop(PropKey.ROLE)
            if not role:
                raise ValueError(f"missing role in user '{participant.name}'")
            if role not in DEFINED_ROLES:
                print(f"Warning: '{role}' of {participant.name} is not a defined role {DEFINED_ROLES}")

        participants = self.participants_by_types.get(participant.type)
        if not participants:
            participants = []
            self.participants_by_types[participant.type] = participants
        participants.append(participant)
        self.all_names[participant.name] = participant
        return participant

    def add_client(self, name: str, org: str, props: dict):
        return self.add_participant(Participant(ParticipantType.CLIENT, name, org, props))

    def get_clients(self):
        return self.get_all_participants(ParticipantType.CLIENT)

    def add_relay(self, name: str, org: str, props: dict):
        return self.add_participant(Participant(ParticipantType.RELAY, name, org, props))

    def get_relays(self):
        return self.get_all_participants(ParticipantType.RELAY)

    def add_admin(self, name: str, org: str, props: dict):
        return self.add_participant(Participant(ParticipantType.ADMIN, name, org, props))

    def get_admins(self):
        return self.get_all_participants(ParticipantType.ADMIN)

    def get_all_participants(self, types: Union[None, str, List[str]] = None):
        if not types:
            # get all types
            return list(self.all_names.values())

        if isinstance(types, str):
            types = [types]
        elif not isinstance(types, list):
            raise ValueError(f"types must be a str or List[str] but got {type(types)}")

        result = []
        processed_types = []  # in case 'types' contains duplicates
        for t in types:
            if t not in processed_types:
                ps = self.participants_by_types.get(t)
                if ps:
                    result.extend(ps)
                processed_types.append(t)
        return result

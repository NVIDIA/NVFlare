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

    def __str__(self):
        return f"Entity[{self.name=}, {self.props=}, {self.parent=}]"

    def __repr__(self):
        return self.__str__()


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
            err, reason = name_check(type, "simple_name")
            if err:
                raise ValueError(reason)
            print(f"Warning: participant type '{type}' of {name} is not a defined type {DEFINED_PARTICIPANT_TYPES}")

        err, reason = name_check(org, "org")
        if err:
            raise ValueError(reason)

        if type == ParticipantType.ADMIN:
            if not props:
                raise ValueError(f"missing role for admin '{name}'")

            role = props.get(PropKey.ROLE)
            if not role:
                raise ValueError(f"missing role for admin '{name}'")

            err, reason = name_check(role, "simple_name")
            if err:
                raise ValueError(f"bad role value '{role}' for admin '{name}': {reason}")

            if role not in DEFINED_ROLES:
                print(f"Warning: '{role}' of admin '{name}' is not a defined role {DEFINED_ROLES}")

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
        """Get listening host property of the participant

        Returns: a ListeningHost object, or None if the property is not defined.

        """
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
        """Get the connect_to property of the participant

        Returns: a ConnectTo object

        """
        h = self.get_prop(PropKey.CONNECT_TO)
        if not h:
            return None
        else:
            return parse_connect_to(h)


def _must_get(d: dict, key: str):
    """Must get property of the specified key from the dict

    Args:
        d: the dict that contains participant properties
        key: key of the property to get

    Returns: the value of the property. If the property does not exist, ValueError exception is raised.

    """
    v = d.pop(key, None)
    if not v:
        raise ValueError(f"missing participant {key}")
    return v


def participant_from_dict(participant_def: dict) -> Participant:
    """Create a Participant from a dict that contains participant property definitions.

    Args:
        participant_def: the dict that contains participant definition

    Returns: a Participant object

    """
    if not isinstance(participant_def, dict):
        raise ValueError(f"participant_def must be dict but got {type(participant_def)}")

    name = _must_get(participant_def, PropKey.NAME)
    t = _must_get(participant_def, PropKey.TYPE)
    org = _must_get(participant_def, PropKey.ORG)
    return Participant(type=t, name=name, org=org, props=participant_def)


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
        self._participants_by_types = {}  # participant type => list of participants
        self._all_names = {}  # name => participant

        if participants:
            if not isinstance(participants, list):
                raise ValueError(f"participants must be a list of Participant but got {type(participants)}")

            for p in participants:
                if not isinstance(p, Participant):
                    raise ValueError(f"bad item in participants: must be Participant but got {type(p)}")
                self.add_participant(p)

    def set_server(self, name: str, org: str, props: dict) -> Participant:
        """Set the server of the project.

        Args:
            name: name of the server.
            org: org of the server
            props: additional server properties.

        Returns: a Participant object for the server

        """
        return self.add_participant(Participant(ParticipantType.SERVER, name, org, props))

    def get_server(self) -> Optional[Participant]:
        """Get the server definition. Only one server is supported!

        Returns: server participant

        """
        return self.server

    def get_overseer(self) -> Optional[Participant]:
        """Get the overseer definition.

        Note: overseer is deprecated.

        Returns: None

        """
        return None

    def add_participant(self, participant: Participant) -> Participant:
        """Add a participant to the project.
        Before adding the participant, this method checks the following conditions:
        - All participants in the project must have unique names
        - Only one server is allowed in the project
        - Role must be specified for admin type of participant

        Args:
            participant: the participant to be added.

        Returns: the participant object added.

        """
        if participant.name in self._all_names:
            raise ValueError(f"the project {self.name} already has a participant with the name '{participant.name}'")

        participant.parent = self
        if participant.type == ParticipantType.SERVER:
            if self.server:
                raise ValueError(f"cannot add participant {participant.name} as server - server already exists")
            self.server = participant
        elif participant.type == ParticipantType.OVERSEER:
            raise ValueError(f"cannot add participant {participant.name} as overseer - overseer is removed")

        participants = self._participants_by_types.get(participant.type)
        if not participants:
            participants = []
            self._participants_by_types[participant.type] = participants
        participants.append(participant)
        self._all_names[participant.name] = participant
        return participant

    def add_client(self, name: str, org: str, props: dict) -> Participant:
        """Add a client to the project

        Args:
            name: name of the client
            org: org of the client
            props: additional properties of the client

        Returns: the Participant object of the client

        """
        return self.add_participant(Participant(ParticipantType.CLIENT, name, org, props))

    def get_clients(self) -> List[Participant]:
        """Get all clients of the project

        Returns: a list of clients

        """
        return self.get_all_participants(ParticipantType.CLIENT)

    def add_relay(self, name: str, org: str, props: dict) -> Participant:
        """Add a relay to the project

        Args:
            name: name of the relay
            org: org of the relay
            props: additional properties of the relay

        Returns: the relay Participant object

        """
        return self.add_participant(Participant(ParticipantType.RELAY, name, org, props))

    def get_relays(self) -> List[Participant]:
        """Get all relays of the project

        Returns: the list of relays of the project

        """
        return self.get_all_participants(ParticipantType.RELAY)

    def add_admin(self, name: str, org: str, props: dict) -> Participant:
        """Add an admin user to the project

        Args:
            name: name of the admin user.
            org: org of the admin user.
            props: properties of the user definition

        Returns: a Participant object of the admin user

        """
        return self.add_participant(Participant(ParticipantType.ADMIN, name, org, props))

    def get_admins(self) -> List[Participant]:
        """Get the list of admin users

        Returns: list of admin users

        """
        return self.get_all_participants(ParticipantType.ADMIN)

    def get_all_participants(self, types: Union[None, str, List[str]] = None):
        """Get all participants of the project of specified types.

        Args:
            types: types of the participants to be returned.

        Returns: all participants of the project of specified types.
            If 'types' is not specified (None), it returns all participants of the project;
            If 'types' is a str, it is treated as a single type and participants of this type is returned;
            If 'types' is a list of types, participants of these types are returned;

        """
        if not types:
            # get all types
            return list(self._all_names.values())

        if isinstance(types, str):
            types = [types]
        elif not isinstance(types, list):
            raise ValueError(f"types must be a str or List[str] but got {type(types)}")

        result = []
        processed_types = []  # in case 'types' contains duplicates
        for t in types:
            if t not in processed_types:
                ps = self._participants_by_types.get(t)
                if ps:
                    result.extend(ps)
                processed_types.append(t)
        return result

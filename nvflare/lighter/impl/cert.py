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

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import NameOID

from nvflare.lighter.constants import CertFileBasename, CtxKey, ParticipantType, PropKey
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Participant, Project
from nvflare.lighter.spec import Builder
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key


class _CertState:

    CERT_STATE_FILE = "cert.json"

    PROP_ROOT_CERT = CtxKey.ROOT_CERT
    PROP_ROOT_PRI_KEY = CtxKey.ROOT_PRI_KEY
    PROP_CERT = "cert"
    PROP_PRI_KEY = "pri_key"

    def __init__(self, state_dir: str):
        self.is_available = False
        self.state_dir = state_dir
        self.content = {}
        cert_file = os.path.join(state_dir, self.CERT_STATE_FILE)
        if os.path.exists(cert_file):
            self.is_available = True
            with open(cert_file, "rt") as f:
                self.content.update(json.load(f))

    def get_root_cert(self):
        return self.content.get(self.PROP_ROOT_CERT)

    def set_root_cert(self, cert):
        self.content[self.PROP_ROOT_CERT] = cert

    def get_root_pri_key(self):
        return self.content.get(self.PROP_ROOT_PRI_KEY)

    def set_root_pri_key(self, key):
        self.content[self.PROP_ROOT_PRI_KEY] = key

    def has_subject(self, subject: str):
        return subject in self.content

    def _add_subject_prop(self, subject: str, key: str, value):
        subject_data = self.content.get(subject)
        if not subject_data:
            subject_data = {}
            self.content[subject] = subject_data
        subject_data[key] = value

    def _get_subject_prop(self, subject: str, key: str):
        subject_data = self.content.get(subject)
        if not subject_data:
            return None
        return subject_data.get(key)

    def add_subject_cert(self, subject: str, cert):
        self._add_subject_prop(subject, self.PROP_CERT, cert)

    def get_subject_cert(self, subject: str):
        return self._get_subject_prop(subject, self.PROP_CERT)

    def add_subject_pri_key(self, subject: str, pri_key):
        self._add_subject_prop(subject, self.PROP_PRI_KEY, pri_key)

    def get_subject_pri_key(self, subject: str):
        return self._get_subject_prop(subject, self.PROP_PRI_KEY)

    def persist(self):
        cert_file = os.path.join(self.state_dir, self.CERT_STATE_FILE)
        with open(cert_file, "wt") as f:
            json.dump(self.content, f)


class CertBuilder(Builder):
    def __init__(self):
        """Build certificate chain for every participant.

        Handles building (creating and self-signing) the root CA certificates, creating server, client and
        admin certificates, and having them signed by the root CA for secure communication. If the state folder has
        information about previously generated certs, it loads them back and reuses them.
        """
        self.root_cert = None
        self.persistent_state = None
        self.serialized_cert = None
        self.pri_key = None
        self.pub_key = None
        self.subject = None
        self.issuer = None

    def initialize(self, project: Project, ctx: ProvisionContext):
        state_dir = ctx.get_state_dir()
        self.persistent_state = _CertState(state_dir)
        state = self.persistent_state

        if project.root_private_key:
            # using project provided credentials
            self.serialized_cert = project.serialized_root_cert
            self.root_cert = x509.load_pem_x509_certificate(self.serialized_cert, default_backend())
            self.pri_key = project.root_private_key
            self.pub_key = self.pri_key.public_key()
            self.subject = self.root_cert.subject
            self.issuer = self.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
            state.is_available = True
        elif state.is_available:
            state_root_cert = state.get_root_cert()
            self.serialized_cert = state_root_cert.encode("ascii")
            self.root_cert = x509.load_pem_x509_certificate(self.serialized_cert, default_backend())

            state_pri_key = state.get_root_pri_key()
            self.pri_key = serialization.load_pem_private_key(
                state_pri_key.encode("ascii"), password=None, backend=default_backend()
            )

            self.pub_key = self.pri_key.public_key()
            self.subject = self.root_cert.subject
            self.issuer = self.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value

    def _build_root(self, subject, subject_org):
        assert isinstance(self.persistent_state, _CertState)
        if not self.persistent_state.is_available:
            pri_key, pub_key = generate_keys()
            self.issuer = subject
            self.root_cert = self._generate_cert(subject, subject_org, self.issuer, pri_key, pub_key, ca=True)
            self.pri_key = pri_key
            self.pub_key = pub_key
            self.serialized_cert = serialize_cert(self.root_cert)

            self.persistent_state.set_root_cert(self.serialized_cert.decode("ascii"))
            self.persistent_state.set_root_pri_key(serialize_pri_key(self.pri_key).decode("ascii"))

    def _build_write_cert_pair(self, participant: Participant, base_name, ctx: ProvisionContext):
        assert isinstance(self.persistent_state, _CertState)
        subject = participant.subject
        if self.persistent_state.has_subject(subject):
            subject_cert = self.persistent_state.get_subject_cert(subject)
            cert = x509.load_pem_x509_certificate(subject_cert.encode("ascii"), default_backend())

            subject_pri_key = self.persistent_state.get_subject_pri_key(subject)
            pri_key = serialization.load_pem_private_key(
                subject_pri_key.encode("ascii"), password=None, backend=default_backend()
            )

            if participant.type == ParticipantType.ADMIN:
                cn_list = cert.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
                for cn in cn_list:
                    role = cn.value
                    new_role = participant.get_prop(PropKey.ROLE)
                    if role != new_role:
                        err_msg = (
                            f"{participant.name}'s previous role is {role} but is now {new_role}.\n"
                            + "Please delete existing workspace and provision from scratch."
                        )
                        raise RuntimeError(err_msg)
        else:
            pri_key, cert = self.get_pri_key_cert(participant)
            self.persistent_state.add_subject_cert(subject, serialize_cert(cert).decode("ascii"))
            self.persistent_state.add_subject_pri_key(subject, serialize_pri_key(pri_key).decode("ascii"))

        dest_dir = ctx.get_kit_dir(participant)
        with open(os.path.join(dest_dir, f"{base_name}.crt"), "wb") as f:
            f.write(serialize_cert(cert))
        with open(os.path.join(dest_dir, f"{base_name}.key"), "wb") as f:
            f.write(serialize_pri_key(pri_key))

        if participant.type in [ParticipantType.CLIENT, ParticipantType.RELAY]:
            self._build_internal_listener_cert(participant, ctx)

        with open(os.path.join(dest_dir, "rootCA.pem"), "wb") as f:
            f.write(self.serialized_cert)

    def _build_internal_listener_cert(self, participant: Participant, ctx: ProvisionContext):
        """Build server cert if the participant has internal listeners.
        Note that internal listener used to be only used for connecting SJ to SP, and CJ to SP, but now
        relay hierarchy is connected to internal listeners.

        Just like the FL Server, a relay could offer one or more hosts for other relays and clients to
        connect to. Therefore, the relay's server cert must include all these host names and IP addresses
        for others to make SSL-based connections using any one of these host names/addresses.

        Args:
            participant: the participant being provisioned
            ctx: a ProvisionContext object

        Returns: None

        """
        lh = participant.get_listening_host()
        if not lh:
            return

        dest_dir = ctx.get_kit_dir(participant)
        project = ctx.get_project()

        # make a fake/temp server participant to use the get_pri_key_cert() method!
        tmp_participant = Participant(
            type=ParticipantType.SERVER,
            name=participant.name,
            org=participant.org,
            project=project,
            props={
                PropKey.HOST_NAMES: lh.host_names,
                PropKey.DEFAULT_HOST: lh.default_host,
            },
        )
        tmp_pri_key, tmp_cert = self.get_pri_key_cert(tmp_participant)

        # The listener cert is a Server Cert.
        bn = CertFileBasename.SERVER
        with open(os.path.join(dest_dir, f"{bn}.crt"), "wb") as f:
            f.write(serialize_cert(tmp_cert))
        with open(os.path.join(dest_dir, f"{bn}.key"), "wb") as f:
            f.write(serialize_pri_key(tmp_pri_key))

    def build(self, project: Project, ctx: ProvisionContext):
        self._build_root(project.name, subject_org=None)
        ctx[CtxKey.ROOT_CERT] = self.root_cert
        ctx[CtxKey.ROOT_PRI_KEY] = self.pri_key

        overseer = project.get_overseer()
        if overseer:
            self._build_write_cert_pair(overseer, CertFileBasename.OVERSEER, ctx)

        server = project.get_server()
        if server:
            self._build_write_cert_pair(server, CertFileBasename.SERVER, ctx)

        for client in project.get_clients():
            self._build_write_cert_pair(client, CertFileBasename.CLIENT, ctx)

        for relay in project.get_relays():
            self._build_write_cert_pair(relay, CertFileBasename.CLIENT, ctx)

        for admin in project.get_admins():
            self._build_write_cert_pair(admin, CertFileBasename.CLIENT, ctx)

    def get_pri_key_cert(self, participant: Participant):
        pri_key, pub_key = generate_keys()
        subject = participant.subject
        subject_org = participant.org
        if participant.type == ParticipantType.ADMIN:
            role = participant.get_prop(PropKey.ROLE)
        else:
            role = None

        server = participant if participant.type == ParticipantType.SERVER else None
        cert = self._generate_cert(
            subject,
            subject_org,
            self.issuer,
            self.pri_key,
            pub_key,
            role=role,
            server=server,
        )
        return pri_key, cert

    @staticmethod
    def _generate_cert(
        subject,
        subject_org,
        issuer,
        signing_pri_key,
        subject_pub_key,
        valid_days=360,
        ca=False,
        role=None,
        server: Participant = None,
    ):
        server_default_host = None
        server_additional_hosts = None

        if server:
            # This is to generate a server cert.
            # Use SubjectAlternativeName for all host names
            server_default_host = server.get_default_host()
            server_additional_hosts = server.get_prop(PropKey.HOST_NAMES)

        return generate_cert(
            subject=Identity(subject, subject_org, role),
            issuer=Identity(issuer),
            signing_pri_key=signing_pri_key,
            subject_pub_key=subject_pub_key,
            valid_days=valid_days,
            ca=ca,
            server_default_host=server_default_host,
            server_additional_hosts=server_additional_hosts,
        )

    def finalize(self, project: Project, ctx: ProvisionContext):
        assert isinstance(self.persistent_state, _CertState)
        self.persistent_state.persist()

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

from nvflare.lighter import utils
from nvflare.lighter.constants import (
    CommConfigArg,
    ConnSecurity,
    CtxKey,
    ParticipantType,
    PropKey,
    ProvFileName,
    ProvisionMode,
    TemplateSectionKey,
)
from nvflare.lighter.entity import Participant
from nvflare.lighter.spec import Builder, Project, ProvisionContext


class StaticFileBuilder(Builder):
    def __init__(
        self,
        config_folder="",
        scheme="grpc",
        app_validator="",
        download_job_url="",
        docker_image="",
        **kwargs,
    ):
        """Build all static files from template.

        Uses the information from project.yml through project to go through the participants and write the contents of
        each file with the template, and replacing with the appropriate values from project.yml.

        Usually, two main categories of files are created in all FL participants, static and dynamic. Static files
        have similar contents among different participants, with small differences.  For example, the differences in
        sub_start.sh are client name and python module.  Those are basically static files.  This builder uses template
        file and string replacement to generate those static files for each participant.

        Args:
            config_folder: usually "config"
            app_validator: optional path to an app validator to verify that uploaded app has the expected structure
            docker_image: when docker_image is set to a docker image name, docker.sh will be generated on
            server/client/admin
        """
        if not isinstance(scheme, str):
            raise ValueError(f"invalid scheme: must be str but got {type(scheme)}")
        scheme = scheme.lower().strip()
        if not scheme:
            raise ValueError("scheme is not specified")

        builtin_schemes = ["grpc", "tcp", "http"]
        if scheme not in builtin_schemes:
            # we only issue warning since it could be a custom scheme
            print(f"WARNING: {scheme} is not a builtin scheme {builtin_schemes}")
        self.config_folder = config_folder
        self.scheme = scheme
        self.docker_image = docker_image
        self.download_job_url = download_job_url
        self.app_validator = app_validator
        self.aio_schemes = {
            "tcp": "atcp",
            "grpc": "agrpc",
        }

    @staticmethod
    def _build_conn_properties(site: Participant, ctx: ProvisionContext):
        valid_values = [ConnSecurity.CLEAR, ConnSecurity.TLS, ConnSecurity.MTLS]
        conn_security = site.get_prop_fb(PropKey.CONN_SECURITY)
        if conn_security:
            assert isinstance(conn_security, str)
            conn_security = conn_security.lower()

            if conn_security not in valid_values:
                raise ValueError(f"invalid connection_security '{conn_security}': must be in {valid_values}")
        else:
            conn_security = ConnSecurity.MTLS

        custom_ca_cert = site.get_prop_fb(PropKey.CUSTOM_CA_CERT)
        if custom_ca_cert:
            shutil.copyfile(custom_ca_cert, os.path.join(ctx.get_kit_dir(site), ProvFileName.CUSTOM_CA_CERT_FILE_NAME))
        return conn_security

    def _determine_scheme(self, participant: Participant, scheme=None) -> str:
        use_aio = participant.get_prop(PropKey.USE_AIO, False)
        if not scheme:
            scheme = self.scheme

        if use_aio:
            scheme = self.aio_schemes.get(scheme)
            if not scheme:
                scheme = self.scheme
        return scheme

    def _build_server(self, server: Participant, ctx: ProvisionContext):
        project = ctx.get_project()
        dest_dir = ctx.get_kit_dir(server)

        admin_port = ctx.get(CtxKey.ADMIN_PORT)
        fed_learn_port = ctx.get(CtxKey.FED_LEARN_PORT)
        target = f"{server.name}:{fed_learn_port}"
        sp_end_point = f"{server.name}:{fed_learn_port}:{admin_port}"
        conn_sec = self._build_conn_properties(server, ctx)

        ctx.build_from_template(
            dest_dir,
            TemplateSectionKey.FED_SERVER,
            ProvFileName.FED_SERVER_JSON,
            replacement={
                "name": project.name,
                "target": target,
                "admin_server": server.name,
                "admin_port": admin_port,
                "scheme": self._determine_scheme(server),
                "conn_sec": conn_sec,
                "sp_end_point": sp_end_point,
            },
        )

        self._build_comm_config_for_internal_listener(server)

        replacement_dict = {
            "admin_port": admin_port,
            "fed_learn_port": fed_learn_port,
            "config_folder": self.config_folder,
            "docker_image": self.docker_image,
            "org_name": server.org,
            "type": "server",
            "app_name": "server_train",
            "cln_uid": "",
        }

        if self.docker_image:
            ctx.build_from_template(
                dest_dir,
                TemplateSectionKey.DOCKER_SERVER_SH,
                ProvFileName.DOCKER_SH,
                replacement=replacement_dict,
                exe=True,
            )

        ctx.build_from_template(
            dest_dir,
            TemplateSectionKey.START_SERVER_SH,
            ProvFileName.START_SH,
            replacement={"ha_mode": "false"},
            exe=True,
        )

        ctx.build_from_template(
            dest_dir,
            TemplateSectionKey.SUB_START_SH,
            ProvFileName.SUB_START_SH,
            replacement=replacement_dict,
            exe=True,
        )

        ctx.build_from_template(dest_dir, TemplateSectionKey.STOP_FL_SH, ProvFileName.STOP_FL_SH, exe=True)

        # local folder creation
        dest_dir = ctx.get_local_dir(server)

        ctx.build_from_template(dest_dir, TemplateSectionKey.LOG_CONFIG, ProvFileName.LOG_CONFIG_DEFAULT, exe=False)

        ctx.build_from_template(
            dest_dir, TemplateSectionKey.LOCAL_SERVER_RESOURCES, ProvFileName.RESOURCES_JSON_DEFAULT, exe=False
        )

        ctx.build_from_template(
            dest_dir, TemplateSectionKey.SAMPLE_PRIVACY, ProvFileName.PRIVACY_JSON_SAMPLE, exe=False
        )

        # other builder (e.g. CC) can set the AUTHZ_SECTION_KEY to specify authorization policies for the server
        authz_section_key = server.get_prop(PropKey.AUTHZ_SECTION_KEY, TemplateSectionKey.DEFAULT_AUTHZ)

        ctx.build_from_template(dest_dir, authz_section_key, ProvFileName.AUTHORIZATION_JSON_DEFAULT, exe=False)

        # workspace folder file
        dest_dir = ctx.get_ws_dir(server)
        ctx.build_from_template(dest_dir, TemplateSectionKey.SERVER_README, ProvFileName.README_TXT, exe=False)

    def _build_comm_config_for_internal_listener(self, participant: Participant):
        """Build template args for comm_config, which will be used to create internal listener

        Args:
            participant:the participant that will create internal listener

        Returns: None

        Note: we only build template args but do not build the comm_config.json here. This is because
        comm_config.json contains multiple sections that are built in different places. The creation of
        comm_config.json happens during "finalize" of this builder, which will apply all template args
        to the "comm_config" template.

        """
        lh = participant.get_listening_host()
        if not lh:
            return

        scheme = lh.scheme
        if participant.type == ParticipantType.RELAY:
            scheme = self._determine_scheme(participant, scheme)

        replacement_dict = {
            CommConfigArg.SCHEME: scheme,
            CommConfigArg.HOST: lh.default_host,
            CommConfigArg.PORT: lh.port,
            CommConfigArg.CONN_SEC: lh.conn_sec,
        }

        comm_config_args = participant.get_prop(PropKey.COMM_CONFIG_ARGS)
        comm_config_args.update(replacement_dict)

    def _build_client(self, client: Participant, ctx: ProvisionContext):
        project = ctx.get_project()
        server = project.get_server()
        if not server:
            raise ValueError("missing server definition in project")

        dest_dir = ctx.get_kit_dir(client)

        is_leaf = client.get_prop(PropKey.IS_LEAF, True)
        if is_leaf:
            is_leaf = "true"
        else:
            is_leaf = "false"

        admin_port = ctx.get(CtxKey.ADMIN_PORT)
        fl_port = ctx.get(CtxKey.FED_LEARN_PORT)
        conn_host, conn_port = self._determine_conn_target(client, ctx)
        if conn_port:
            fl_port = conn_port
        sp_end_point = f"{conn_host}:{fl_port}:{admin_port}"

        ctx.build_from_template(
            dest_dir,
            TemplateSectionKey.FED_CLIENT,
            ProvFileName.FED_CLIENT_JSON,
            replacement={
                "scheme": self._determine_scheme(client),
                "name": project.name,
                "server_identity": server.name,
                "fqsn": client.get_prop(PropKey.FQSN),
                "is_leaf": is_leaf,
                "conn_sec": self._build_conn_properties(client, ctx),
                "sp_end_point": sp_end_point,
            },
        )

        # build internal comm
        self._build_comm_config_for_internal_listener(client)

        replacement_dict = {
            "admin_port": admin_port,
            "fed_learn_port": fl_port,
            "client_name": f"{client.subject}",
            "config_folder": self.config_folder,
            "docker_image": self.docker_image,
            "org_name": client.org,
            "type": "client",
            "app_name": "client_train",
            "cln_uid": f"uid={client.subject}",
        }

        if self.docker_image:
            ctx.build_from_template(
                dest_dir,
                TemplateSectionKey.DOCKER_CLIENT_SH,
                ProvFileName.DOCKER_SH,
                replacement_dict,
                exe=True,
            )

        ctx.build_from_template(dest_dir, TemplateSectionKey.START_CLIENT_SH, ProvFileName.START_SH, exe=True)

        ctx.build_from_template(
            dest_dir, TemplateSectionKey.SUB_START_SH, ProvFileName.SUB_START_SH, replacement_dict, exe=True
        )

        ctx.build_from_template(dest_dir, TemplateSectionKey.STOP_FL_SH, ProvFileName.STOP_FL_SH, exe=True)

        # local folder creation
        dest_dir = ctx.get_local_dir(client)

        ctx.build_from_template(dest_dir, TemplateSectionKey.LOG_CONFIG, ProvFileName.LOG_CONFIG_DEFAULT)

        num_gpus = 0
        gpu_mem = 0
        capacity = client.get_prop(PropKey.CAPACITY)
        if capacity:
            num_gpus = capacity.get(PropKey.NUM_GPUS, 0)
            gpu_mem = capacity.get(PropKey.GPU_MEM, 0)

        replacement_dict = {
            "num_gpus": num_gpus,
            "gpu_mem": gpu_mem,
        }

        ctx.build_from_template(
            dest_dir,
            TemplateSectionKey.LOCAL_CLIENT_RESOURCES,
            ProvFileName.RESOURCES_JSON_DEFAULT,
            replacement=replacement_dict,
            content_modify_cb=self._modify_error_sender,
            client=client,
        )

        ctx.build_from_template(
            dest_dir,
            TemplateSectionKey.SAMPLE_PRIVACY,
            ProvFileName.PRIVACY_JSON_SAMPLE,
        )

        # other builder (e.g. CC) can set the AUTHZ_SECTION_KEY to specify authorization policies for this client
        authz_section_key = client.get_prop(PropKey.AUTHZ_SECTION_KEY, TemplateSectionKey.DEFAULT_AUTHZ)

        ctx.build_from_template(dest_dir, authz_section_key, ProvFileName.AUTHORIZATION_JSON_DEFAULT)

        # build relay__resources if relay is used by this client
        ct = client.get_connect_to()
        if ct and ct.name and ct.name != server.name:
            # relay is used!
            relay_map = ctx.get(CtxKey.RELAY_MAP)
            if not relay_map:
                raise RuntimeError(f"missing {CtxKey.RELAY_MAP} from the provision context")

            relay = relay_map.get(ct.name)
            if not relay:
                raise RuntimeError(f"cannot find relay {ct.name} from the map")

            assert isinstance(relay, Participant)
            fqcn = relay.get_prop(PropKey.FQCN)
            if not relay:
                raise RuntimeError(f"cannot find FQCN from relay {ct.name}")

            lh = relay.get_listening_host()
            if not lh:
                raise RuntimeError(f"missing listening_host in relay {relay.name}")

            host = ct.host
            if not host:
                # use the relay's default host
                host = lh.default_host
            else:
                # validate against the relay's host names
                err = self._validate_host_name_against_listener(
                    host_name=host,
                    listener_name=relay.name,
                    listener_default_host=lh.default_host,
                    listener_available_host_names=lh.host_names,
                )
                if err:
                    ctx.warning(f"the connect_to.host '{host}' in client {client.name} may be invalid: {err}")

            scheme = lh.scheme
            if not scheme:
                scheme = "grpc"

            conn_sec = ct.conn_sec
            if not conn_sec:
                conn_sec = lh.conn_sec
            if not conn_sec:
                conn_sec = "mtls"

            relay_port = ct.port
            if not relay_port:
                relay_port = lh.port

            addr = f"{host}:{relay_port}"

            replacement_dict = {
                "scheme": scheme,
                "identity": relay.name,
                "address": addr,
                "fqcn": fqcn,
                "conn_sec": conn_sec,
            }

            ctx.build_from_template(
                dest_dir,
                TemplateSectionKey.RELAY_RESOURCES_JSON,
                ProvFileName.RELAY_RESOURCES_JSON,
                replacement_dict,
                exe=False,
            )

            # build comm config backbone gen
            replacement_dict = {CommConfigArg.CONN_GEN: 1}
            comm_config_args = client.get_prop(PropKey.COMM_CONFIG_ARGS)
            comm_config_args.update(replacement_dict)

        # workspace folder file
        dest_dir = ctx.get_ws_dir(client)
        ctx.build_from_template(dest_dir, TemplateSectionKey.CLIENT_README, ProvFileName.README_TXT)

    def _modify_error_sender(self, section: str, client: Participant) -> str:
        """Modify the local resources section and remove the "error_log_sender" component if necessary.
        By default, the "error_log_sender" component is included in local resources.
        However, if the project does not allow errors to be sent, then this component must be removed.

        Args:
            section: the local resources section generated from template
            client: the client being provisioned

        Returns: modified section content

        """
        allow = client.get_prop_fb(PropKey.ALLOW_ERROR_SENDING, False)
        if allow:
            # error sending is allowed - so no change needed.
            return section

        # convert to dict for easy modification
        section_dict = json.loads(section)
        components = section_dict.get("components")
        if not components:
            return section

        assert isinstance(components, list)
        for c in components:
            if c["id"] == "error_log_sender":
                # must remove this component
                components.remove(c)
                break

        # Must convert to Json string
        return json.dumps(section_dict, indent=2)

    @staticmethod
    def _check_host_name_against_server(host_name: str, server: Participant) -> str:
        return StaticFileBuilder._validate_host_name_against_listener(
            host_name,
            listener_name=server.name,
            listener_default_host=server.get_default_host(),
            listener_available_host_names=server.get_prop(PropKey.HOST_NAMES),
        )

    @staticmethod
    def _validate_host_name_against_listener(
        host_name: str, listener_name: str, listener_default_host: str, listener_available_host_names: list
    ) -> str:
        """Validate specified host_name against default host and available host names of the listener.
        This is to make sure that host_name used by a connector is valid.

        Args:
            host_name: the host name to be validated
            listener_name: name of the listener
            listener_default_host: the default host of the listener
            listener_available_host_names: other available host names of the listener

        Returns: error message if any

        """
        if host_name == listener_default_host:
            # Use the default host - OK
            return ""

        if listener_available_host_names and host_name in listener_available_host_names:
            # use alternative host name - OK
            return ""

        return f"host name '{host_name}' is not defined in '{listener_name}'"

    def _determine_conn_target(self, participant, ctx: ProvisionContext):
        project = ctx.get_project()
        server = project.get_server()
        if not server:
            raise ValueError(f"Missing server definition in project {project.name}")

        port = None
        ct = participant.get_connect_to()
        if ct:
            if not ct.name or ct.name == server.name:
                # connect to server directly - no relay
                err = self._check_host_name_against_server(ct.host, server)
                if err:
                    ctx.warning(f"connect_to.host '{ct.host}' in {participant.subject} may be invalid: {err}")
                conn_host = ct.host
            else:
                # uses relay
                # since relay config is done in relay__resources.json, host name doesn't matter here.
                # we use default server here
                conn_host = server.get_default_host()

            if ct.port:
                port = ct.port
        else:
            # connect_to is not explicitly specified: use the server's name by default
            # Note: by doing this dynamically, we guarantee the sp_end_point to be correct, even if the
            # project.yaml does not specify the default server host correctly!
            conn_host = server.get_default_host()
        return conn_host, port

    def _build_admin(self, admin: Participant, ctx: ProvisionContext):
        self.prepare_admin_config(admin, ctx)

        dest_dir = ctx.get_kit_dir(admin)
        admin_port = ctx.get(CtxKey.ADMIN_PORT)
        server_name = ctx.get(CtxKey.SERVER_NAME)

        replacement_dict = {
            "admin_name": admin.name,
            "cn": f"{server_name}",
            "admin_port": f"{admin_port}",
            "docker_image": self.docker_image,
        }

        if self.docker_image:
            ctx.build_from_template(
                dest_dir, TemplateSectionKey.DOCKER_ADMIN_SH, ProvFileName.DOCKER_SH, replacement_dict, exe=True
            )

        ctx.build_from_template(
            dest_dir,
            TemplateSectionKey.FL_ADMIN_SH,
            ProvFileName.FL_ADMIN_SH,
            replacement=replacement_dict,
            exe=True,
        )

        ctx.build_from_template(
            dest_dir,
            TemplateSectionKey.ADM_NOTEBOOK,
            ProvFileName.SYSTEM_INFO_IPYNB,
            replacement=replacement_dict,
        )

        ctx.build_from_template(dest_dir, TemplateSectionKey.ADMIN_README, ProvFileName.README_TXT)

    def prepare_admin_config(self, admin: Participant, ctx: ProvisionContext):
        project = ctx.get_project()
        server = project.get_server()
        conn_sec = admin.get_prop_fb(PropKey.CONN_SECURITY)
        if not conn_sec:
            conn_sec = ConnSecurity.MTLS

        uid_source = "user_input"
        provision_mode = ctx.get_provision_mode()
        if provision_mode == ProvisionMode.POC:
            uid_source = "cert"

        conn_host, conn_port = self._determine_conn_target(admin, ctx)
        if not conn_port:
            conn_port = ctx.get(CtxKey.ADMIN_PORT)

        replacement_dict = {
            "project_name": project.name,
            "username": "" if provision_mode == ProvisionMode.POC else admin.name,
            "server_identity": server.name,
            "scheme": self.scheme,
            "conn_sec": conn_sec,
            "host": conn_host,
            "port": conn_port,
            "uid_source": uid_source,
        }
        ctx.build_from_template(
            dest_dir=ctx.get_kit_dir(admin),
            temp_section=TemplateSectionKey.FED_ADMIN,
            file_name=ProvFileName.FED_ADMIN_JSON,
            replacement=replacement_dict,
        )

        # create default resources in local
        ctx.build_from_template(
            dest_dir=ctx.get_local_dir(admin),
            temp_section=TemplateSectionKey.DEFAULT_ADMIN_RESOURCES,
            file_name=ProvFileName.RESOURCES_JSON_DEFAULT,
        )

    def _build_relay(self, relay: Participant, ctx: ProvisionContext):
        lh = relay.get_listening_host()
        if not lh:
            # Relay must have listening_host defined!
            raise ValueError(f"missing listening_host in relay {relay.name}")

        # build comm config backbone gen - relay's backbone_conn_gen is always 1, so it won't
        # connect to the server.
        # Note: the default gen is 2, we change to 1 for relay.
        replacement_dict = {CommConfigArg.CONN_GEN: 1}
        comm_config_args = relay.get_prop(PropKey.COMM_CONFIG_ARGS)
        comm_config_args.update(replacement_dict)
        self._build_comm_config_for_internal_listener(relay)

        # build fed_relay.json
        project = ctx.get_project()
        assert isinstance(project, Project)
        server = project.get_server()
        assert isinstance(server, Participant)

        # default parent is server
        parent_scheme = self.scheme
        parent_identity = server.name
        parent_fqcn = "server"
        parent_host = server.get_default_host()
        parent_port = ctx.get(CtxKey.FED_LEARN_PORT)
        parent_conn_sec = server.get_prop_fb(PropKey.CONN_SECURITY)

        ct = relay.get_connect_to()
        lh = None
        parent_relay = None
        if ct:
            if ct.name:
                parent_identity = ct.name
                if ct.name != server.name:
                    # use relay
                    relay_map = ctx.get(CtxKey.RELAY_MAP)
                    parent_relay = relay_map.get(ct.name)
                    if not parent_relay:
                        # this should never happen since the map has been validated
                        raise RuntimeError(f"cannot get parent relay {ct.name} for relay {relay.name}")

                    assert isinstance(parent_relay, Participant)
                    lh = parent_relay.get_listening_host()
                    parent_scheme = lh.scheme
                    parent_fqcn = parent_relay.get_prop(PropKey.FQCN)

                    # check whether the specified ct.host is available from the parent relay
                    if ct.host:
                        err = self._validate_host_name_against_listener(
                            host_name=ct.host,
                            listener_name=parent_relay.name,
                            listener_default_host=lh.default_host,
                            listener_available_host_names=lh.host_names,
                        )
                        if err:
                            # even though ct.host is not available from the parent relay, we do not
                            # treat it as a hard error since the customer may intentionally connect to
                            # a different host (BYOConn).
                            ctx.warning(f"the connect_to.host '{ct.host}' in relay {relay.name} may be invalid: {err}")

            # general logic: properties defined in connect_to overrides parent's listening_host.
            if ct.host:
                parent_host = ct.host
            elif lh:
                parent_host = lh.default_host

            if ct.port:
                parent_port = ct.port
            elif lh:
                parent_port = lh.port

            if ct.conn_sec:
                parent_conn_sec = ct.conn_sec
            elif lh:
                parent_conn_sec = lh.conn_sec

            if not parent_relay and ct.host:
                # no parent relay - directly connect to server
                # if host is specified explicitly in connect_to, check whether the specified host
                # is available from the server.
                err = self._check_host_name_against_server(ct.host, server)
                if err:
                    # the host specified in connect_to is not available from the server.
                    # we do not treat it as a hard error because in the case of BYOConn, the customer
                    # may intentionally connect the relay to another host.
                    ctx.warning(f"the connect_to.host '{ct.host}' in relay {relay.name} may be invalid: {err}")

        parent_addr = f"{parent_host}:{parent_port}"
        replacement_dict = {
            "project_name": project.name,
            "identity": relay.name,
            "server_identity": server.name,
            "scheme": parent_scheme,
            "parent_identity": parent_identity,
            "address": parent_addr,
            "fqcn": parent_fqcn,
            "conn_sec": parent_conn_sec,
        }

        dest_dir = ctx.get_kit_dir(relay)
        ctx.build_from_template(
            dest_dir=dest_dir,
            file_name=ProvFileName.FED_RELAY_JSON,
            temp_section=TemplateSectionKey.FED_RELAY,
            replacement=replacement_dict,
        )

        replacement_dict = {
            "config_folder": self.config_folder,
            "org_name": relay.org,
            "type": "relay",
            "app_name": "relay",
            "cln_uid": f"uid={relay.subject}",
        }

        ctx.build_from_template(dest_dir, TemplateSectionKey.START_CLIENT_SH, ProvFileName.START_SH, exe=True)

        ctx.build_from_template(
            dest_dir, TemplateSectionKey.SUB_START_SH, ProvFileName.SUB_START_SH, replacement_dict, exe=True
        )

        ctx.build_from_template(dest_dir, TemplateSectionKey.STOP_FL_SH, ProvFileName.STOP_FL_SH, exe=True)

        # other local resources
        dest_dir = ctx.get_local_dir(relay)
        ctx.build_from_template(dest_dir, TemplateSectionKey.LOG_CONFIG, ProvFileName.LOG_CONFIG_DEFAULT)

    def build(self, project: Project, ctx: ProvisionContext):
        server = project.get_server()
        if server:
            self._build_server(server, ctx)

        for relay in project.get_relays():
            self._build_relay(relay, ctx)

        for client in project.get_clients():
            self._build_client(client, ctx)

        for admin in project.get_admins():
            self._build_admin(admin, ctx)

    @staticmethod
    def _determine_relay_hierarchy(project: Project, ctx: ProvisionContext):
        """Relays are organized hierarchically. Relay hierarchy must be determined before we can generate
        their FQCNs properly. This method determines relay hierarchy based on the connect_to properties
        in all specified relays. Circular refs are not allowed. FQCN for each relay is determined.

        Args:
            project: the project being provisioned
            ctx: the ProvisionContext object

        Returns:

        """
        # Build name => relay map
        name_to_relay = {}

        relays = project.get_relays()
        if not relays:
            # nothing to prepare
            return

        for r in relays:
            assert isinstance(r, Participant)
            name_to_relay[r.name] = r

        # determine relay parents based connect_to
        server = project.get_server()
        for r in relays:
            assert isinstance(r, Participant)
            ct = r.get_connect_to()

            if ct and ct.name and ct.name != server.name:
                # parent is another relay
                parent = name_to_relay.get(ct.name)
                if not parent:
                    raise ValueError(f"undefined parent {ct.name} in relay {r.name}")
            else:
                # parent is the server
                parent = None  # None parent represents server
            r.set_prop(PropKey.PARENT, parent)

        # determine FQCNs
        for r in relays:
            fqcn_path = []
            err = check_parent(r, fqcn_path)
            if err:
                raise ValueError(f"bad relay definitions: {err}")
            fqcn = ".".join(fqcn_path)
            r.set_prop(PropKey.FQCN, fqcn)

        if name_to_relay:
            ctx[CtxKey.RELAY_MAP] = name_to_relay

    @staticmethod
    def _determine_client_hierarchy(project: Project, ctx: ProvisionContext):
        """Client hierarchy is used to enable hierarchical FL algorithms.
        This method determines client hierarchy based on the "parent" property.
        FQSN (fully qualified site name) defines the position of the client in the hierarchy.
        FQSN is computed for each client.

        Args:
            project: the project being provisioned
            ctx: a ProvisionContext object

        Returns:

        """
        # Build name => client map
        client_map = {}

        clients = project.get_clients()
        if not clients:
            # nothing to prepare
            return

        for c in clients:
            assert isinstance(c, Participant)
            client_map[c.name] = c

        # determine client parents
        server = project.get_server()
        for c in clients:
            assert isinstance(c, Participant)
            parent_name = c.get_prop(PropKey.PARENT)
            parent_client = None  # parent is server by default
            if parent_name and parent_name != server.name:
                # parent is another client
                parent_client = client_map.get(parent_name)
                if not parent_client:
                    raise ValueError(f"undefined parent client '{parent_name}' in client {c.name}")
            c.set_prop(PropKey.PARENT, parent_client)

        # determine FQSNs (fully qualified site name)
        for c in clients:
            fqsn_path = []
            err = check_parent(c, fqsn_path)
            if err:
                raise ValueError(f"bad client definitions: {err}")
            fqsn = ".".join(fqsn_path)
            c.set_prop(PropKey.FQSN, fqsn)

        if client_map:
            ctx[CtxKey.CLIENT_MAP] = client_map

    def initialize(self, project: Project, ctx: ProvisionContext):
        ctx.load_templates("master_template.yml")
        self._determine_relay_hierarchy(project, ctx)
        self._determine_client_hierarchy(project, ctx)

        # prepare clients comm config
        for p in project.get_all_participants():
            assert isinstance(p, Participant)
            p.set_prop(PropKey.COMM_CONFIG_ARGS, {})

    def finalize(self, project: Project, ctx: ProvisionContext):
        for p in project.get_all_participants():
            assert isinstance(p, Participant)
            comm_config_args = p.get_prop(PropKey.COMM_CONFIG_ARGS)
            if comm_config_args:
                # create comm config file
                # we create the comm_config here because multiple builders may create different portions of the file.
                assert isinstance(comm_config_args, dict)

                replacement_dict = {
                    CommConfigArg.CONN_GEN: 2,
                    CommConfigArg.PORT: 0,  # meaning undefined
                    CommConfigArg.HOST: "localhost",
                    CommConfigArg.SCHEME: "tcp",
                    CommConfigArg.CONN_SEC: ConnSecurity.CLEAR,
                }

                replacement_dict.update(comm_config_args)

                ctx.build_from_template(
                    dest_dir=ctx.get_local_dir(p),
                    temp_section=TemplateSectionKey.COMM_CONFIG,
                    file_name=ProvFileName.COMM_CONFIG,
                    replacement=replacement_dict,
                    content_modify_cb=_remove_undefined_port,
                )

        # create start_all.sh
        gen_scripts = project.props.get("gen_scripts", False)
        if gen_scripts:
            self._create_start_all(project, ctx)

    @staticmethod
    def _append(content: str, participant: Participant) -> str:
        return content + f"$BASE_DIR/{participant.name}/startup/start.sh\n"

    def _create_start_all(self, project: Project, ctx: ProvisionContext):
        """Create the start_all.sh script to be used for starting all sites (server, relays and clients).
        This is a convenience script and not part of any site's startup kit.

        Args:
            project: project being provisioned
            ctx: a ProvisionContext object

        Returns: None

        """
        content = '#!/usr/bin/env bash\nBASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        server = project.get_server()
        content = self._append(content, server)

        # include all relays
        relays = project.get_relays()
        if relays:
            # sort relays based on their FQCNs
            relays.sort(key=lambda x: len(x.get_prop(PropKey.FQCN)))
            for r in relays:
                content = self._append(content, r)

        # include all clients
        for c in project.get_clients():
            content = self._append(content, c)

        utils.write(os.path.join(ctx.get_wip_dir(), ProvFileName.START_ALL_SH), content, "t", exe=True)


def _remove_undefined_port(section: str) -> str:
    """This is the callback for checking and removing undefined port number for comm_config.
    Since the templating system does not allow conditional args, each arg must have a value when
    generating the section from the template. We used port 0 to represent undefined port number.
    We must remove undefined port number from comm_config; otherwise Flare wouldn't work in run time.

    Args:
        section: the section data to be checked

    Returns: modified section data

    """
    # section is JSON string - convert to dict for easy check and modification
    section_dict = json.loads(section)
    resources = section_dict.get("internal", {}).get("resources")
    if resources:
        port = resources.get(PropKey.PORT)
        if port is None or port == 0:
            # this is undefined port - remove it
            resources.pop(PropKey.PORT, None)

        # convert dict back to JSON string
        return json.dumps(section_dict, indent=2)
    else:
        # no change
        return section


def check_parent(c: Participant, path: list):
    if c.name in path:
        return f"circular parent ref {c.name}"

    path.insert(0, c.name)
    parent = c.get_prop(PropKey.PARENT)
    if not parent:
        return ""
    assert isinstance(parent, Participant)
    parent.set_prop(PropKey.IS_LEAF, False)
    return check_parent(parent, path)

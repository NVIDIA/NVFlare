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

import copy
import json
import os
import shutil

import yaml

from nvflare.lighter import utils
from nvflare.lighter.constants import (
    ConnSecurity,
    CtxKey,
    OverseerRole,
    PropKey,
    ProvFileName,
    ProvisionMode,
    TemplateSectionKey,
)
from nvflare.lighter.entity import Participant, parse_connect_to
from nvflare.lighter.spec import Builder, Project, ProvisionContext


class StaticFileBuilder(Builder):
    def __init__(
        self,
        config_folder="",
        scheme="grpc",
        app_validator="",
        download_job_url="",
        docker_image="",
        overseer_agent: dict = None,
        components="",
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
        self.config_folder = config_folder
        self.scheme = scheme
        self.docker_image = docker_image
        self.download_job_url = download_job_url
        self.app_validator = app_validator
        self.overseer_agent = overseer_agent
        self.components = components

    def _build_overseer(self, overseer: Participant, ctx: ProvisionContext):
        dest_dir = ctx.get_kit_dir(overseer)
        protocol = overseer.get_prop(PropKey.PROTOCOL, "http")
        api_root = overseer.get_prop(PropKey.API_ROOT, "/api/v1/")
        default_port = "443" if protocol == "https" else "80"
        port = overseer.get_prop(PropKey.PORT, default_port)
        replacement_dict = {"port": port, "hostname": overseer.name}

        project = ctx.get_project()
        admins = project.get_admins()
        privilege_dict = dict()
        for admin in admins:
            role = admin.get_prop(PropKey.ROLE)
            if role in privilege_dict:
                privilege_dict[role].append(admin.subject)
            else:
                privilege_dict[role] = [admin.subject]

        utils.write(
            os.path.join(dest_dir, ProvFileName.PRIVILEGE_YML),
            yaml.dump(privilege_dict, Dumper=yaml.Dumper),
            "t",
            exe=False,
        )

        if self.docker_image:
            ctx.build_from_template(
                dest_dir, TemplateSectionKey.DOCKER_SERVER_SH, ProvFileName.DOCKER_SH, replacement_dict, exe=True
            )

        ctx.build_from_template(
            dest_dir,
            TemplateSectionKey.GUNICORN_CONF_PY,
            ProvFileName.GUNICORN_CONF_PY,
            replacement_dict,
            exe=False,
        )

        ctx.build_from_template(dest_dir, TemplateSectionKey.START_OVERSEER_SH, ProvFileName.START_SH, exe=True)

        if port:
            ctx[PropKey.OVERSEER_END_POINT] = f"{protocol}://{overseer.name}:{port}{api_root}"
        else:
            ctx[PropKey.OVERSEER_END_POINT] = f"{protocol}://{overseer.name}{api_root}"

    @staticmethod
    def _build_conn_properties(site: Participant, ctx: ProvisionContext, site_config: dict):
        valid_values = [ConnSecurity.CLEAR, ConnSecurity.TLS, ConnSecurity.MTLS]
        conn_security = site.get_prop_fb(PropKey.CONN_SECURITY)
        if conn_security:
            assert isinstance(conn_security, str)
            conn_security = conn_security.lower()

            if conn_security not in valid_values:
                raise ValueError(f"invalid connection_security '{conn_security}': must be in {valid_values}")

            site_config["connection_security"] = conn_security

        custom_ca_cert = site.get_prop_fb(PropKey.CUSTOM_CA_CERT)
        if custom_ca_cert:
            shutil.copyfile(custom_ca_cert, os.path.join(ctx.get_kit_dir(site), ProvFileName.CUSTOM_CA_CERT_FILE_NAME))

    def _build_server(self, server: Participant, ctx: ProvisionContext):
        project = ctx.get_project()
        config = ctx.json_load_template_section(TemplateSectionKey.FED_SERVER)
        dest_dir = ctx.get_kit_dir(server)
        server_0 = config["servers"][0]
        server_0["name"] = project.name
        admin_port = ctx.get(CtxKey.ADMIN_PORT)
        fed_learn_port = ctx.get(CtxKey.FED_LEARN_PORT)
        communication_port = server.get_prop(CtxKey.DOCKER_COMM_PORT)
        server_0["service"]["target"] = f"{server.name}:{fed_learn_port}"
        server_0["service"]["scheme"] = self.scheme
        server_0["admin_host"] = server.name
        server_0["admin_port"] = admin_port

        self._prepare_overseer_agent(server, config, OverseerRole.SERVER, ctx)

        # set up connection props
        self._build_conn_properties(server, ctx, server_0)

        utils.write(os.path.join(dest_dir, ProvFileName.FED_SERVER_JSON), json.dumps(config, indent=2), "t")

        replacement_dict = {
            "admin_port": admin_port,
            "fed_learn_port": fed_learn_port,
            "communication_port": communication_port,
            "config_folder": self.config_folder,
            "docker_image": self.docker_image,
            "org_name": server.org,
            "type": "server",
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

        ctx.build_from_template(dest_dir, TemplateSectionKey.START_SERVER_SH, ProvFileName.START_SH, exe=True)

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

        ctx.build_from_template(
            dest_dir, TemplateSectionKey.DEFAULT_AUTHZ, ProvFileName.AUTHORIZATION_JSON_DEFAULT, exe=False
        )

        # workspace folder file
        dest_dir = ctx.get_ws_dir(server)
        ctx.build_from_template(dest_dir, TemplateSectionKey.SERVER_README, ProvFileName.README_TXT, exe=False)

    def _build_client(self, client: Participant, ctx):
        project = ctx.get_project()
        server = project.get_server()
        if not server:
            raise ValueError("missing server definition in project")
        config = ctx.json_load_template_section(TemplateSectionKey.FED_CLIENT)
        dest_dir = ctx.get_kit_dir(client)
        config["servers"][0]["service"]["scheme"] = self.scheme
        config["servers"][0]["name"] = project.name
        config["servers"][0]["identity"] = server.name  # the official identity of the server
        admin_port = ctx.get(CtxKey.ADMIN_PORT)
        fed_learn_port = ctx.get(CtxKey.FED_LEARN_PORT)
        communication_port = client.get_prop(PropKey.DOCKER_COMM_PORT)
        replacement_dict = {
            "admin_port": admin_port,
            "fed_learn_port": fed_learn_port,
            "communication_port": communication_port,
            "comm_host_name": client.name + "-parent",
            "client_name": f"{client.subject}",
            "config_folder": self.config_folder,
            "docker_image": self.docker_image,
            "org_name": client.org,
            "type": "client",
            "cln_uid": f"uid={client.subject}",
        }

        self._prepare_overseer_agent(client, config, OverseerRole.CLIENT, ctx)

        # set connection properties
        client_conf = config["client"]
        self._build_conn_properties(client, ctx, client_conf)

        utils.write(os.path.join(dest_dir, ProvFileName.FED_CLIENT_JSON), json.dumps(config, indent=2), "t")

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

        ctx.build_from_template(
            dest_dir,
            TemplateSectionKey.LOCAL_CLIENT_RESOURCES,
            ProvFileName.RESOURCES_JSON_DEFAULT,
            content_modify_cb=self._modify_error_sender,
            client=client,
        )

        ctx.build_from_template(
            dest_dir,
            TemplateSectionKey.SAMPLE_PRIVACY,
            ProvFileName.PRIVACY_JSON_SAMPLE,
        )

        ctx.build_from_template(dest_dir, TemplateSectionKey.DEFAULT_AUTHZ, ProvFileName.AUTHORIZATION_JSON_DEFAULT)

        # build relay__resources if relay is used by this client
        connect_to = client.get_prop(PropKey.CONNECT_TO)
        if connect_to:
            name, host, port = parse_connect_to(connect_to)
            if name:
                # relay is used!
                relay_map = ctx.get(CtxKey.RELAY_MAP)
                if not relay_map:
                    raise RuntimeError(f"missing {CtxKey.RELAY_MAP} from the provision context")

                relay = relay_map.get(name)
                if not relay:
                    raise RuntimeError(f"cannot find relay {name} from the map")

                assert isinstance(relay, Participant)
                fqcn = relay.get_prop(PropKey.FQCN)
                if not relay:
                    raise RuntimeError(f"cannot find FQCN from relay {name}")

                relay_port = relay.get_prop_fb(PropKey.RELAY_PORT)
                if not relay_port:
                    raise RuntimeError(f"cannot find relay_port from relay {name}")

                if not host:
                    # use the relay's default host
                    host = relay.get_default_host()

                scheme = relay.get_prop(PropKey.SCHEME, "grpc")
                conn_sec = relay.get_prop(PropKey.CONN_SECURITY, "mtls")
                if port:
                    relay_port = port

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
                replacement_dict = {"conn_gen": 1}
                section = ctx.build_section_from_template(
                    temp_section=TemplateSectionKey.COMM_CONFIG_BACKBONE_GEN, replacement=replacement_dict
                )
                client.add_prop(PropKey.COMM_CONFIG_GEN, section)

        # workspace folder file
        dest_dir = ctx.get_ws_dir(client)
        ctx.build_from_template(dest_dir, TemplateSectionKey.CLIENT_README, ProvFileName.README_TXT)

    def _modify_error_sender(self, section: dict, client: Participant):
        if not isinstance(section, dict):
            return section
        allow = client.get_prop_fb(PropKey.ALLOW_ERROR_SENDING, False)
        if not allow:
            components = section.get("components")
            assert isinstance(components, list)
            for c in components:
                if c["id"] == "error_log_sender":
                    components.remove(c)
                    break

        return section

    @staticmethod
    def _check_host_name(host_name: str, server: Participant) -> str:
        if host_name == server.get_default_host():
            # Use the default host - OK
            return ""

        available_host_names = server.get_prop(PropKey.HOST_NAMES)
        if available_host_names and host_name in available_host_names:
            # use alternative host name - OK
            return ""

        return f"unknown host name '{host_name}'"

    def _prepare_overseer_agent(self, participant, config, role, ctx: ProvisionContext):
        project = ctx.get_project()
        server = project.get_server()
        if not server:
            raise ValueError(f"Missing server definition in project {project.name}")

        # The properties CtxKey.FED_LEARN_PORT and CtxKey.ADMIN_PORT are guaranteed to exist
        fl_port = ctx.get(CtxKey.FED_LEARN_PORT)
        admin_port = ctx.get(CtxKey.ADMIN_PORT)

        if self.overseer_agent:
            overseer_agent = copy.deepcopy(self.overseer_agent)
            if overseer_agent.get("overseer_exists", True):
                if role == OverseerRole.SERVER:
                    overseer_agent["args"] = {
                        "role": role,
                        "overseer_end_point": ctx.get("overseer_end_point", ""),
                        "project": project.name,
                        "name": server.name,
                        "fl_port": str(fl_port),
                        "admin_port": str(admin_port),
                    }
                else:
                    overseer_agent["args"] = {
                        "role": role,
                        "overseer_end_point": ctx.get("overseer_end_point", ""),
                        "project": project.name,
                        "name": participant.subject,
                    }
            else:
                # do not use overseer system
                # Dummy overseer agent is used here
                if role == OverseerRole.SERVER:
                    # the server expects the "connect_to" to be the same as its name
                    # otherwise the host name generated by the dummy agent won't be accepted!
                    connect_to = server.name
                else:
                    connect_to = participant.get_prop(PropKey.CONNECT_TO)
                    if connect_to:
                        name, addr, port = parse_connect_to(connect_to)
                        if not name:
                            err = self._check_host_name(addr, server)
                            if err:
                                ctx.warning(f"connect_to in {participant.subject} may be invalid: {err}")
                            connect_to = addr
                        else:
                            # uses relay
                            # since relay config is done in relay__resources.json, we use default server here
                            connect_to = server.get_default_host()

                        if port:
                            # override fl_port
                            fl_port = port
                    else:
                        # connect_to is not explicitly specified: use the server's name by default
                        # Note: by doing this dynamically, we guarantee the sp_end_point to be correct, even if the
                        # project.yaml does not specify the default server host correctly!
                        connect_to = server.get_default_host()

                # change the sp_end_point to use connect_to
                agent_args = overseer_agent.get("args")
                if agent_args:
                    sp_end_point = agent_args.get("sp_end_point")
                    if sp_end_point:
                        # format of the sp_end_point:  server_host_name:fl_port:admin_port
                        agent_args["sp_end_point"] = f"{connect_to}:{fl_port}:{admin_port}"

            overseer_agent.pop("overseer_exists", None)
            config["overseer_agent"] = overseer_agent

    def _build_admin(self, admin: Participant, ctx: ProvisionContext):
        dest_dir = ctx.get_kit_dir(admin)
        admin_port = ctx.get(CtxKey.ADMIN_PORT)
        server_name = ctx.get(CtxKey.SERVER_NAME)

        replacement_dict = {
            "cn": f"{server_name}",
            "admin_port": f"{admin_port}",
            "docker_image": self.docker_image,
        }

        config = self.prepare_admin_config(admin, ctx)

        utils.write(os.path.join(dest_dir, ProvFileName.FED_ADMIN_JSON), json.dumps(config, indent=2), "t")

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

        ctx.build_from_template(dest_dir, TemplateSectionKey.ADMIN_README, ProvFileName.README_TXT)

    def prepare_admin_config(self, admin, ctx: ProvisionContext):
        config = ctx.json_load_template_section(TemplateSectionKey.FED_ADMIN)
        agent_config = dict()
        self._prepare_overseer_agent(admin, agent_config, OverseerRole.ADMIN, ctx)
        config["admin"].update(agent_config)

        provision_mode = ctx.get_provision_mode()
        if provision_mode == ProvisionMode.POC:
            # in poc mode, we change to use "local_cert" as the cred_type so that the user won't be
            # prompted for username when starting the admin console
            config["admin"]["username"] = admin.name
            config["admin"]["cred_type"] = "local_cert"
        return config

    def build(self, project: Project, ctx: ProvisionContext):
        overseer = project.get_overseer()
        if overseer:
            self._build_overseer(overseer, ctx)

        server = project.get_server()
        if server:
            self._build_server(server, ctx)

        for client in project.get_clients():
            self._build_client(client, ctx)

        for admin in project.get_admins():
            self._build_admin(admin, ctx)

    def initialize(self, project: Project, ctx: ProvisionContext):
        # name => relay
        name_to_relay = {}

        relays = project.get_relays()
        if not relays:
            # nothing to prepare
            return

        for r in relays:
            assert isinstance(r, Participant)
            name_to_relay[r.name] = r

        # determine parents
        for r in relays:
            assert isinstance(r, Participant)
            parent_def = r.get_prop(PropKey.CONNECT_TO)
            print(f"determine parent for relay {r.name} {parent_def=}")
            if parent_def:
                parent_name, parent_addr, parent_port = parse_connect_to(parent_def)
            else:
                # default connect to server
                parent_name, parent_addr, parent_port = None, None, None

            if not parent_name:
                parent = None
            else:
                parent = name_to_relay.get(parent_name)
                if not parent:
                    raise ValueError(f"undefined parent {parent_name} in relay {r.name}")
            r.add_prop(PropKey.PARENT, (parent, parent_addr, parent_port))

        # determine FQCNs
        for r in relays:
            fqcn_path = []
            err = check_parent(r, fqcn_path)
            if err:
                raise ValueError(f"bad relay definitions: {err}")
            fqcn = ".".join(fqcn_path)
            r.add_prop(PropKey.FQCN, fqcn)

        if name_to_relay:
            ctx[CtxKey.RELAY_MAP] = name_to_relay

    def finalize(self, project: Project, ctx: ProvisionContext):
        for p in project.get_all_participants():
            assert isinstance(p, Participant)
            comm_config = {}
            internal = p.get_prop(PropKey.COMM_CONFIG_INTERNAL, "")
            if internal:
                int_dict = json.loads(internal)
                comm_config.update(int_dict)

            gen = p.get_prop(PropKey.COMM_CONFIG_GEN, "")
            if gen:
                gen_dict = json.loads(gen)
                comm_config.update(gen_dict)

            if comm_config:
                # create comm config file
                # we create the comm_config here because multiple builders may create different portions of the file.
                dest_dir = ctx.get_local_dir(p)
                with open(os.path.join(dest_dir, ProvFileName.COMM_CONFIG), "w") as f:
                    json.dump(comm_config, f, indent=2)


def check_parent(c: Participant, path: list):
    if c.name in path:
        return f"circular parent ref {c.name}"

    path.insert(0, c.name)
    parent, _, _ = c.get_prop(PropKey.PARENT)
    if not parent:
        return ""
    return check_parent(parent, path)

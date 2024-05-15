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

import io
import json
import os
import subprocess
import tempfile

from nvflare.lighter import tplt_utils, utils

from .cert import CertPair, Entity, deserialize_ca_key, make_cert
from .models import Client, Project, User

lighter_folder = os.path.dirname(utils.__file__)
template = utils.load_yaml(os.path.join(lighter_folder, "impl", "master_template.yml"))
supported_csps = ["aws", "azure"]
for csp in supported_csps:
    csp_template_file = os.path.join(lighter_folder, "impl", f"{csp}_template.yml")
    if os.path.exists(csp_template_file):
        template.update(utils.load_yaml(csp_template_file))


def get_csp_start_script_name(csp):
    return f"{csp}_start.sh"


def gen_overseer(key):
    project = Project.query.first()
    entity = Entity(project.overseer)
    issuer = Entity(project.short_name)
    signing_cert_pair = CertPair(issuer, project.root_key, project.root_cert)
    cert_pair = make_cert(entity, signing_cert_pair)
    with tempfile.TemporaryDirectory() as tmp_dir:
        overseer_dir = os.path.join(tmp_dir, entity.name)
        dest_dir = os.path.join(overseer_dir, "startup")
        os.mkdir(overseer_dir)
        os.mkdir(dest_dir)
        utils._write(
            os.path.join(dest_dir, "start.sh"),
            template["start_ovsr_sh"],
            "t",
            exe=True,
        )
        utils._write(
            os.path.join(dest_dir, "gunicorn.conf.py"),
            utils.sh_replace(template["gunicorn_conf_py"], {"port": "8443"}),
            "t",
            exe=False,
        )
        utils._write_pki(type="overseer", dest_dir=dest_dir, cert_pair=cert_pair, root_cert=project.root_cert)
        run_args = ["zip", "-rq", "-P", key, "tmp.zip", "."]
        subprocess.run(run_args, cwd=tmp_dir)
        fileobj = io.BytesIO()
        with open(os.path.join(tmp_dir, "tmp.zip"), "rb") as fo:
            fileobj.write(fo.read())
        fileobj.seek(0)
    return fileobj, f"{entity.name}.zip"


def gen_server(key, first_server=True):
    project = Project.query.first()
    if first_server:
        entity = Entity(project.server1)
        fl_port = 8002
        admin_port = 8003
    else:
        entity = Entity(project.server2)
        fl_port = 8102
        admin_port = 8103
    issuer = Entity(project.short_name)
    signing_cert_pair = CertPair(issuer, project.root_key, project.root_cert)
    cert_pair = make_cert(entity, signing_cert_pair)

    config = json.loads(template["fed_server"])
    server_0 = config["servers"][0]
    server_0["name"] = project.short_name
    server_0["service"]["target"] = f"{entity.name}:{fl_port}"
    server_0["service"]["scheme"] = project.scheme if hasattr(project, "scheme") else "grpc"
    server_0["admin_host"] = entity.name
    server_0["admin_port"] = admin_port
    if project.ha_mode:
        overseer_agent = {"path": "nvflare.ha.overseer_agent.HttpOverseerAgent"}
        overseer_agent["args"] = {
            "role": "server",
            "overseer_end_point": f"https://{project.overseer}:8443/api/v1",
            "project": project.short_name,
            "name": entity.name,
            "fl_port": str(fl_port),
            "admin_port": str(admin_port),
        }
    else:
        overseer_agent = {"path": "nvflare.ha.dummy_overseer_agent.DummyOverseerAgent"}
        overseer_agent["args"] = {"sp_end_point": f"{project.server1}:8002:8003"}

    config["overseer_agent"] = overseer_agent
    replacement_dict = {
        "admin_port": admin_port,
        "fed_learn_port": fl_port,
        "config_folder": "config",
        "ha_mode": "true" if project.ha_mode else "false",
        "docker_image": project.app_location.split(" ")[-1] if project.app_location else "nvflare/nvflare",
        "org_name": "",
        "type": "server",
        "cln_uid": "",
    }
    tplt = tplt_utils.Template(template)
    with tempfile.TemporaryDirectory() as tmp_dir:
        server_dir = os.path.join(tmp_dir, entity.name)
        dest_dir = os.path.join(server_dir, "startup")
        os.mkdir(server_dir)
        os.mkdir(dest_dir)
        utils._write_common(
            type="server",
            dest_dir=dest_dir,
            template=template,
            tplt=tplt,
            replacement_dict=replacement_dict,
            config=config,
        )
        utils._write_pki(type="server", dest_dir=dest_dir, cert_pair=cert_pair, root_cert=project.root_cert)
        if not project.ha_mode:
            for csp in supported_csps:
                utils._write(
                    os.path.join(dest_dir, get_csp_start_script_name(csp)),
                    tplt.get_start_sh(csp=csp, type="server", entity=entity),
                    "t",
                    exe=True,
                )
        signatures = utils.sign_all(dest_dir, deserialize_ca_key(project.root_key))
        json.dump(signatures, open(os.path.join(dest_dir, "signature.json"), "wt"))

        # local folder creation
        dest_dir = os.path.join(server_dir, "local")
        os.mkdir(dest_dir)
        utils._write_local(type="server", dest_dir=dest_dir, template=template)

        # workspace folder file
        utils._write(
            os.path.join(server_dir, "readme.txt"),
            template["readme_fs"],
            "t",
        )
        run_args = ["zip", "-rq", "-P", key, "tmp.zip", "."]
        subprocess.run(run_args, cwd=tmp_dir)
        fileobj = io.BytesIO()
        with open(os.path.join(tmp_dir, "tmp.zip"), "rb") as fo:
            fileobj.write(fo.read())
        fileobj.seek(0)
    return fileobj, f"{entity.name}.zip"


def gen_client(key, id):
    project = Project.query.first()
    client = Client.query.get(id)
    entity = Entity(client.name, client.organization.name)
    issuer = Entity(project.short_name)
    signing_cert_pair = CertPair(issuer, project.root_key, project.root_cert)
    cert_pair = make_cert(entity, signing_cert_pair)

    config = json.loads(template["fed_client"])
    config["servers"][0]["name"] = project.short_name
    config["servers"][0]["service"]["scheme"] = project.scheme if hasattr(project, "scheme") else "grpc"
    replacement_dict = {
        "client_name": entity.name,
        "config_folder": "config",
        "docker_image": project.app_location.split(" ")[-1] if project.app_location else "nvflare/nvflare",
        "org_name": entity.org,
        "type": "client",
        "cln_uid": f"uid={entity.name}",
    }
    for k in ["client_name", "org_name", "cln_uid"]:
        value = replacement_dict[k]
        escaped_value = value.replace("'", "\\'")
        replacement_dict[k] = escaped_value

    if project.ha_mode:
        overseer_agent = {"path": "nvflare.ha.overseer_agent.HttpOverseerAgent"}
        overseer_agent["args"] = {
            "role": "client",
            "overseer_end_point": f"https://{project.overseer}:8443/api/v1",
            "project": project.short_name,
            "name": entity.name,
        }
    else:
        overseer_agent = {"path": "nvflare.ha.dummy_overseer_agent.DummyOverseerAgent"}
        overseer_agent["args"] = {"sp_end_point": f"{project.server1}:8002:8003"}
    config["overseer_agent"] = overseer_agent

    tplt = tplt_utils.Template(template)
    with tempfile.TemporaryDirectory() as tmp_dir:
        client_dir = os.path.join(tmp_dir, entity.name)
        dest_dir = os.path.join(client_dir, "startup")
        os.mkdir(client_dir)
        os.mkdir(dest_dir)

        utils._write_pki(type="client", dest_dir=dest_dir, cert_pair=cert_pair, root_cert=project.root_cert)
        utils._write_common(
            type="client",
            dest_dir=dest_dir,
            template=template,
            tplt=tplt,
            replacement_dict=replacement_dict,
            config=config,
        )

        for csp in supported_csps:
            utils._write(
                os.path.join(dest_dir, get_csp_start_script_name(csp)),
                tplt.get_start_sh(csp=csp, type="client", entity=entity),
                "t",
                exe=True,
            )

        signatures = utils.sign_all(dest_dir, deserialize_ca_key(project.root_key))
        json.dump(signatures, open(os.path.join(dest_dir, "signature.json"), "wt"))

        # local folder creation
        dest_dir = os.path.join(client_dir, "local")
        os.mkdir(dest_dir)
        utils._write_local(type="client", dest_dir=dest_dir, template=template, capacity=client.capacity.capacity)

        # workspace folder file
        utils._write(
            os.path.join(client_dir, "readme.txt"),
            template["readme_fc"],
            "t",
        )

        run_args = ["zip", "-rq", "-P", key, "tmp.zip", "."]
        subprocess.run(run_args, cwd=tmp_dir)
        fileobj = io.BytesIO()
        with open(os.path.join(tmp_dir, "tmp.zip"), "rb") as fo:
            fileobj.write(fo.read())
        fileobj.seek(0)
    return fileobj, f"{entity.name}.zip"


def gen_user(key, id):
    project = Project.query.first()
    server_name = project.server1
    user = User.query.get(id)
    entity = Entity(user.email, user.organization.name, user.role.name)
    issuer = Entity(project.short_name)
    signing_cert_pair = CertPair(issuer, project.root_key, project.root_cert)
    cert_pair = make_cert(entity, signing_cert_pair)

    config = json.loads(template["fed_admin"])
    replacement_dict = {"admin_name": entity.name, "cn": server_name, "admin_port": "8003", "docker_image": ""}

    if project.ha_mode:
        overseer_agent = {"path": "nvflare.ha.overseer_agent.HttpOverseerAgent"}
        overseer_agent["args"] = {
            "role": "admin",
            "overseer_end_point": f"https://{project.overseer}:8443/api/v1",
            "project": project.short_name,
            "name": entity.name,
        }
    else:
        overseer_agent = {"path": "nvflare.ha.dummy_overseer_agent.DummyOverseerAgent"}
        overseer_agent["args"] = {"sp_end_point": f"{project.server1}:8002:8003"}
    config["admin"].update({"overseer_agent": overseer_agent})

    with tempfile.TemporaryDirectory() as tmp_dir:
        user_dir = os.path.join(tmp_dir, entity.name)
        dest_dir = os.path.join(user_dir, "startup")
        os.mkdir(user_dir)
        os.mkdir(dest_dir)

        utils._write(os.path.join(dest_dir, "fed_admin.json"), json.dumps(config, indent=2), "t")
        utils._write(
            os.path.join(dest_dir, "fl_admin.sh"),
            utils.sh_replace(template["fl_admin_sh"], replacement_dict),
            "t",
            exe=True,
        )
        utils._write_pki(type="client", dest_dir=dest_dir, cert_pair=cert_pair, root_cert=project.root_cert)
        signatures = utils.sign_all(dest_dir, deserialize_ca_key(project.root_key))
        json.dump(signatures, open(os.path.join(dest_dir, "signature.json"), "wt"))

        # local folder creation
        dest_dir = os.path.join(user_dir, "local")
        os.mkdir(dest_dir)

        # workspace folder file
        utils._write(
            os.path.join(user_dir, "readme.txt"),
            template["readme_am"],
            "t",
        )
        utils._write(
            os.path.join(user_dir, "system_info.ipynb"),
            utils.sh_replace(template["adm_notebook"], replacement_dict),
            "t",
        )
        run_args = ["zip", "-rq", "-P", key, "tmp.zip", "."]
        subprocess.run(run_args, cwd=tmp_dir)
        fileobj = io.BytesIO()
        with open(os.path.join(tmp_dir, "tmp.zip"), "rb") as fo:
            fileobj.write(fo.read())
        fileobj.seek(0)
    return fileobj, f"{entity.name}.zip"

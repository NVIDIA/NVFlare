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

from nvflare.lighter import utils

from .cert import CertPair, Entity, deserialize_ca_key, make_cert
from .models import Client, Project, User

lighter_folder = os.path.dirname(utils.__file__)
template = utils.load_yaml(os.path.join(lighter_folder, "impl", "master_template.yml"))


def get_csp_template(csp, participant, template):
    return template[f"{csp}_start_{participant}_sh"]


def get_csp_start_script_name(csp):
    return f"{csp}_start.sh"


def _write(file_full_path, content, mode, exe=False):
    mode = mode + "w"
    with open(file_full_path, mode) as f:
        f.write(content)
    if exe:
        os.chmod(file_full_path, 0o755)


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
        _write(
            os.path.join(dest_dir, "start.sh"),
            template["start_ovsr_sh"],
            "t",
            exe=True,
        )
        _write(
            os.path.join(dest_dir, "gunicorn.conf.py"),
            utils.sh_replace(template["gunicorn_conf_py"], {"port": "8443"}),
            "t",
            exe=False,
        )
        _write(os.path.join(dest_dir, "overseer.crt"), cert_pair.ser_cert, "b", exe=False)
        _write(os.path.join(dest_dir, "overseer.key"), cert_pair.ser_pri_key, "b", exe=False)
        _write(os.path.join(dest_dir, "rootCA.pem"), project.root_cert, "b", exe=False)
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
        "org_name": "",
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        server_dir = os.path.join(tmp_dir, entity.name)
        dest_dir = os.path.join(server_dir, "startup")
        os.mkdir(server_dir)
        os.mkdir(dest_dir)
        _write(os.path.join(dest_dir, "fed_server.json"), json.dumps(config, indent=2), "t")
        _write(
            os.path.join(dest_dir, "start.sh"),
            utils.sh_replace(template["start_svr_sh"], replacement_dict),
            "t",
            exe=True,
        )
        _write(
            os.path.join(dest_dir, "sub_start.sh"),
            utils.sh_replace(template["sub_start_svr_sh"], replacement_dict),
            "t",
            exe=True,
        )
        _write(
            os.path.join(dest_dir, "stop_fl.sh"),
            template["stop_fl_sh"],
            "t",
            exe=True,
        )
        _write(os.path.join(dest_dir, "server.crt"), cert_pair.ser_cert, "b", exe=False)
        _write(os.path.join(dest_dir, "server.key"), cert_pair.ser_pri_key, "b", exe=False)
        _write(os.path.join(dest_dir, "rootCA.pem"), project.root_cert, "b", exe=False)
        if not project.ha_mode:
            _write(
                os.path.join(dest_dir, get_csp_start_script_name("azure")),
                utils.sh_replace(get_csp_template("azure", "svr", template), {"server_name": entity.name}),
                "t",
                exe=True,
            )
            _write(
                os.path.join(dest_dir, get_csp_start_script_name("aws")),
                utils.sh_replace(get_csp_template("aws", "svr", template), {"server_name": entity.name}),
                "t",
                exe=True,
            )
        signatures = utils.sign_all(dest_dir, deserialize_ca_key(project.root_key))
        json.dump(signatures, open(os.path.join(dest_dir, "signature.json"), "wt"))

        # local folder creation
        dest_dir = os.path.join(server_dir, "local")
        os.mkdir(dest_dir)
        _write(
            os.path.join(dest_dir, "log.config.default"),
            template["log_config"],
            "t",
        )
        _write(
            os.path.join(dest_dir, "resources.json.default"),
            template["local_server_resources"],
            "t",
        )
        _write(
            os.path.join(dest_dir, "privacy.json.sample"),
            template["sample_privacy"],
            "t",
        )
        _write(
            os.path.join(dest_dir, "authorization.json.default"),
            template["default_authz"],
            "t",
        )

        # workspace folder file
        _write(
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
    replacement_dict = {
        "client_name": entity.name,
        "config_folder": "config",
        "docker_image": "",
        "org_name": entity.org,
    }
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        client_dir = os.path.join(tmp_dir, entity.name)
        dest_dir = os.path.join(client_dir, "startup")
        os.mkdir(client_dir)
        os.mkdir(dest_dir)

        _write(os.path.join(dest_dir, "fed_client.json"), json.dumps(config, indent=2), "t")
        _write(
            os.path.join(dest_dir, "start.sh"),
            template["start_cln_sh"],
            "t",
            exe=True,
        )
        _write(
            os.path.join(dest_dir, "sub_start.sh"),
            utils.sh_replace(template["sub_start_cln_sh"], replacement_dict),
            "t",
            exe=True,
        )
        _write(
            os.path.join(dest_dir, "stop_fl.sh"),
            template["stop_fl_sh"],
            "t",
            exe=True,
        )
        _write(os.path.join(dest_dir, "client.crt"), cert_pair.ser_cert, "b", exe=False)
        _write(os.path.join(dest_dir, "client.key"), cert_pair.ser_pri_key, "b", exe=False)
        _write(os.path.join(dest_dir, "rootCA.pem"), project.root_cert, "b", exe=False)
        _write(
            os.path.join(dest_dir, get_csp_start_script_name("azure")),
            get_csp_template("azure", "cln", template),
            "t",
            exe=True,
        )
        _write(
            os.path.join(dest_dir, get_csp_start_script_name("aws")),
            get_csp_template("aws", "cln", template),
            "t",
            exe=True,
        )
        signatures = utils.sign_all(dest_dir, deserialize_ca_key(project.root_key))
        json.dump(signatures, open(os.path.join(dest_dir, "signature.json"), "wt"))

        # local folder creation
        dest_dir = os.path.join(client_dir, "local")
        os.mkdir(dest_dir)
        _write(
            os.path.join(dest_dir, "log.config.default"),
            template["log_config"],
            "t",
        )
        resources = json.loads(template["local_client_resources"])
        for component in resources["components"]:
            if "nvflare.app_common.resource_managers.gpu_resource_manager.GPUResourceManager" == component["path"]:
                component["args"] = json.loads(client.capacity.capacity)
                break
        _write(
            os.path.join(dest_dir, "resources.json.default"),
            json.dumps(resources, indent=2),
            "t",
        )
        _write(
            os.path.join(dest_dir, "privacy.json.sample"),
            template["sample_privacy"],
            "t",
        )
        _write(
            os.path.join(dest_dir, "authorization.json.default"),
            template["default_authz"],
            "t",
        )
        # workspace folder file
        _write(
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

        _write(os.path.join(dest_dir, "fed_admin.json"), json.dumps(config, indent=2), "t")
        _write(
            os.path.join(dest_dir, "fl_admin.sh"),
            utils.sh_replace(template["fl_admin_sh"], replacement_dict),
            "t",
            exe=True,
        )
        _write(os.path.join(dest_dir, "client.crt"), cert_pair.ser_cert, "b", exe=False)
        _write(os.path.join(dest_dir, "client.key"), cert_pair.ser_pri_key, "b", exe=False)
        _write(os.path.join(dest_dir, "rootCA.pem"), project.root_cert, "b", exe=False)
        signatures = utils.sign_all(dest_dir, deserialize_ca_key(project.root_key))
        json.dump(signatures, open(os.path.join(dest_dir, "signature.json"), "wt"))

        # local folder creation
        dest_dir = os.path.join(user_dir, "local")
        os.mkdir(dest_dir)

        # workspace folder file
        _write(
            os.path.join(user_dir, "readme.txt"),
            template["readme_am"],
            "t",
        )
        _write(
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

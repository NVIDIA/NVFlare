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
import random
import shutil
from base64 import b64decode, b64encode

import yaml
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from nvflare.lighter.impl.cert import load_crt
from nvflare.lighter.tool_consts import NVFLARE_SIG_FILE, NVFLARE_SUBMITTER_CRT_FILE


def generate_password(passlen=16):
    s = "abcdefghijklmnopqrstuvwxyz01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    p = "".join(random.sample(s, passlen))
    return p


def sign_one(content, signing_pri_key):
    signature = signing_pri_key.sign(
        data=content,
        padding=padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        algorithm=hashes.SHA256(),
    )
    return b64encode(signature).decode("utf-8")


def load_private_key_file(file_path):
    with open(file_path, "rt") as f:
        pri_key = serialization.load_pem_private_key(f.read().encode("ascii"), password=None, backend=default_backend())
    return pri_key


def sign_folders(folder, signing_pri_key, crt_path, max_depth=9999):
    depth = 0
    for root, folders, files in os.walk(folder):
        depth = depth + 1
        signatures = dict()
        for file in files:
            if file == NVFLARE_SIG_FILE or file == NVFLARE_SUBMITTER_CRT_FILE:
                continue
            signature = signing_pri_key.sign(
                data=open(os.path.join(root, file), "rb").read(),
                padding=padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                algorithm=hashes.SHA256(),
            )
            signatures[file] = b64encode(signature).decode("utf-8")
        for folder in folders:
            signature = signing_pri_key.sign(
                data=folder.encode("utf-8"),
                padding=padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                algorithm=hashes.SHA256(),
            )
            signatures[folder] = b64encode(signature).decode("utf-8")

        json.dump(signatures, open(os.path.join(root, NVFLARE_SIG_FILE), "wt"))
        shutil.copyfile(crt_path, os.path.join(root, NVFLARE_SUBMITTER_CRT_FILE))
        if depth >= max_depth:
            break


def verify_folder_signature(src_folder, root_ca_path):
    try:
        root_ca_cert = load_crt(root_ca_path)
        root_ca_public_key = root_ca_cert.public_key()
        for root, folders, files in os.walk(src_folder):
            try:
                signatures = json.load(open(os.path.join(root, NVFLARE_SIG_FILE), "rt"))
                cert = load_crt(os.path.join(root, NVFLARE_SUBMITTER_CRT_FILE))
                public_key = cert.public_key()
            except:
                continue  # TODO: shall return False
            root_ca_public_key.verify(
                cert.signature, cert.tbs_certificate_bytes, padding.PKCS1v15(), cert.signature_hash_algorithm
            )
            for k in signatures:
                signatures[k] = b64decode(signatures[k].encode("utf-8"))
            for file in files:
                if file == NVFLARE_SIG_FILE or file == NVFLARE_SUBMITTER_CRT_FILE:
                    continue
                signature = signatures.get(file)
                if signature:
                    public_key.verify(
                        signature=signature,
                        data=open(os.path.join(root, file), "rb").read(),
                        padding=padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                        algorithm=hashes.SHA256(),
                    )
            for folder in folders:
                signature = signatures.get(folder)
                if signature:
                    public_key.verify(
                        signature=signature,
                        data=folder.encode("utf-8"),
                        padding=padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                        algorithm=hashes.SHA256(),
                    )
        return True
    except Exception as e:
        return False


def sign_all(content_folder, signing_pri_key):
    signatures = dict()
    for f in os.listdir(content_folder):
        path = os.path.join(content_folder, f)
        if os.path.isfile(path):
            signature = signing_pri_key.sign(
                data=open(path, "rb").read(),
                padding=padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                algorithm=hashes.SHA256(),
            )
            signatures[f] = b64encode(signature).decode("utf-8")
    return signatures


def load_yaml(file):
    if isinstance(file, str):
        return yaml.safe_load(open(file, "r"))
    elif isinstance(file, bytes):
        return yaml.safe_load(file)
    else:
        return None


def sh_replace(src, mapping_dict):
    result = src
    for k, v in mapping_dict.items():
        result = result.replace("{~~" + k + "~~}", str(v))
    return result


def update_project_server_name_config(project_config: dict, old_server_name, server_name) -> dict:
    update_participant_server_name(project_config, old_server_name, server_name)
    update_overseer_server_name(project_config, old_server_name, server_name)
    return project_config


def update_overseer_server_name(project_config, old_server_name, server_name):
    # update overseer_agent builder
    builders = project_config.get("builders", [])
    for b in builders:
        if "args" in b:
            if "overseer_agent" in b["args"]:
                end_point = b["args"]["overseer_agent"]["args"]["sp_end_point"]
                new_end_point = end_point.replace(old_server_name, server_name)
                b["args"]["overseer_agent"]["args"]["sp_end_point"] = new_end_point


def update_participant_server_name(project_config, old_server_name, new_server_name):
    participants = project_config["participants"]
    for p in participants:
        if p["type"] == "server" and p["name"] == old_server_name:
            p["name"] = new_server_name
            return


def update_project_server_name(project_file: str, old_server_name, server_name):
    with open(project_file, "r") as file:
        project_config = yaml.safe_load(file)

    if not project_config:
        raise RuntimeError("project_config is empty")

    update_project_server_name_config(project_config, old_server_name, server_name)

    with open(project_file, "w") as file:
        yaml.dump(project_config, file)


def update_storage_locations(
    local_dir: str,
    workspace: str,
    default_resource_name: str = "resources.json.default",
    job_storage_name: str = "jobs-storage",
    snapshot_storage_name: str = "snapshot-storage",
):
    """Creates resources.json with snapshot-storage and jobs-storage set as folders directly under the workspace
    for the provided local_dir."""
    default_resource = f"{local_dir}/{default_resource_name}"
    target_resource = f"{local_dir}/resources.json"
    job_storage = f"{workspace}/{job_storage_name}"
    snapshot_storage = f"{workspace}/{snapshot_storage_name}"

    # load resources.json
    with open(default_resource, "r") as f:
        resources = json.load(f)

    # update resources
    resources["snapshot_persistor"]["args"]["storage"]["args"]["root_dir"] = snapshot_storage
    components = resources["components"]
    job_mgr_comp = [comp for comp in components if comp["id"] == "job_manager"][0]
    job_mgr_comp["args"]["uri_root"] = job_storage

    # Serializing json, Writing to resources.json
    json_object = json.dumps(resources, indent=4)
    with open(target_resource, "w") as outfile:
        outfile.write(json_object)


def _write(file_full_path, content, mode, exe=False):
    mode = mode + "w"
    with open(file_full_path, mode) as f:
        f.write(content)
    if exe:
        os.chmod(file_full_path, 0o755)


def _write_common(type, dest_dir, template, tplt, replacement_dict, config):
    mapping = {"server": "svr", "client": "cln"}
    _write(os.path.join(dest_dir, f"fed_{type}.json"), json.dumps(config, indent=2), "t")
    _write(
        os.path.join(dest_dir, "docker.sh"),
        sh_replace(template[f"docker_{mapping[type]}_sh"], replacement_dict),
        "t",
        exe=True,
    )
    _write(
        os.path.join(dest_dir, "start.sh"),
        sh_replace(template[f"start_{mapping[type]}_sh"], replacement_dict),
        "t",
        exe=True,
    )
    _write(
        os.path.join(dest_dir, "sub_start.sh"),
        sh_replace(tplt.get_sub_start_sh(), replacement_dict),
        "t",
        exe=True,
    )
    _write(
        os.path.join(dest_dir, "stop_fl.sh"),
        template["stop_fl_sh"],
        "t",
        exe=True,
    )


def _write_local(type, dest_dir, template, capacity=""):
    _write(
        os.path.join(dest_dir, "log.config.default"),
        template["log_config"],
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
    resources = json.loads(template["local_client_resources"])
    if type == "client":
        for component in resources["components"]:
            if "nvflare.app_common.resource_managers.gpu_resource_manager.GPUResourceManager" == component["path"]:
                component["args"] = json.loads(capacity)
                break
    _write(
        os.path.join(dest_dir, "resources.json.default"),
        json.dumps(resources, indent=2),
        "t",
    )


def _write_pki(type, dest_dir, cert_pair, root_cert):
    _write(os.path.join(dest_dir, f"{type}.crt"), cert_pair.ser_cert, "b", exe=False)
    _write(os.path.join(dest_dir, f"{type}.key"), cert_pair.ser_pri_key, "b", exe=False)
    _write(os.path.join(dest_dir, "rootCA.pem"), root_cert, "b", exe=False)

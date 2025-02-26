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

import datetime
import json
import os
import random
import shutil
from base64 import b64decode, b64encode

import yaml
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import NameOID

from nvflare.lighter.tool_consts import NVFLARE_SIG_FILE, NVFLARE_SUBMITTER_CRT_FILE


class Identity:
    def __init__(self, name: str, org: str = None, role: str = None):
        self.name = name
        self.org = org
        self.role = role


def generate_cert(
    subject: Identity,
    issuer: Identity,
    signing_pri_key,
    subject_pub_key,
    valid_days=360,
    ca=False,
    server_default_host=None,
    server_additional_hosts=None,
):
    if isinstance(server_additional_hosts, str):
        server_additional_hosts = [server_additional_hosts]

    x509_subject = x509_name(subject.name, subject.org, subject.role)
    x509_issuer = x509_name(issuer.name, issuer.org, issuer.role)

    builder = (
        x509.CertificateBuilder()
        .subject_name(x509_subject)
        .issuer_name(x509_issuer)
        .public_key(subject_pub_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=valid_days))
    )
    if ca:
        builder = (
            builder.add_extension(
                x509.SubjectKeyIdentifier.from_public_key(subject_pub_key),
                critical=False,
            )
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(subject_pub_key),
                critical=False,
            )
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=False)
        )

    if server_default_host:
        # This is to generate a server cert.
        # Use SubjectAlternativeName for all host names
        sans = [x509.DNSName(server_default_host)]
        if server_additional_hosts:
            for h in server_additional_hosts:
                if h != server_default_host:
                    sans.append(x509.DNSName(h))
        builder = builder.add_extension(x509.SubjectAlternativeName(sans), critical=False)
    else:
        builder = builder.add_extension(x509.SubjectAlternativeName([x509.DNSName(subject.name)]), critical=False)
    return builder.sign(signing_pri_key, hashes.SHA256(), default_backend())


def serialize_pri_key(pri_key, passphrase=None):
    if passphrase is None or not isinstance(passphrase, bytes):
        return pri_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    else:
        return pri_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.BestAvailableEncryption(password=passphrase),
        )


def serialize_cert(cert):
    return cert.public_bytes(serialization.Encoding.PEM)


def generate_keys():
    pri_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    pub_key = pri_key.public_key()
    return pri_key, pub_key


def x509_name(cn_name, org_name=None, role=None):
    name = [x509.NameAttribute(NameOID.COMMON_NAME, cn_name)]
    if org_name is not None:
        name.append(x509.NameAttribute(NameOID.ORGANIZATION_NAME, org_name))
    if role:
        name.append(x509.NameAttribute(NameOID.UNSTRUCTURED_NAME, role))
    return x509.Name(name)


def load_crt(path):
    with open(path, "rb") as f:
        return load_crt_bytes(f.read())


def load_crt_bytes(data: bytes):
    return x509.load_pem_x509_certificate(data, default_backend())


def generate_password(passlen=16):
    s = "abcdefghijklmnopqrstuvwxyz01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    p = "".join(random.sample(s, passlen))
    return p


def sign_content(content, signing_pri_key, return_str=True):
    if isinstance(content, str):
        content = content.encode("utf-8")  # to bytes
    signature = signing_pri_key.sign(
        data=content,
        padding=_content_padding(),
        algorithm=_content_hash_algo(),
    )

    # signature is bytes
    if return_str:
        return b64encode(signature).decode("utf-8")
    else:
        return signature


def _content_padding():
    return padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH)


def _content_hash_algo():
    return hashes.SHA256()


def verify_content(content, signature, public_key):
    if isinstance(content, str):
        content = content.encode("utf-8")  # to bytes
    if isinstance(signature, str):
        signature = b64decode(signature.encode("utf-8"))  # decode to bytes
    public_key.verify(
        signature=signature,
        data=content,
        padding=_content_padding(),
        algorithm=_content_hash_algo(),
    )


def verify_cert(cert_to_be_verified, root_ca_public_key):
    root_ca_public_key.verify(
        cert_to_be_verified.signature,
        cert_to_be_verified.tbs_certificate_bytes,
        padding.PKCS1v15(),
        cert_to_be_verified.signature_hash_algorithm,
    )


def load_private_key(data: str):
    return serialization.load_pem_private_key(data.encode("ascii"), password=None, backend=default_backend())


def load_private_key_file(file_path):
    with open(file_path, "rt") as f:
        return load_private_key(f.read())


def sign_folders(folder, signing_pri_key, crt_path, max_depth=9999):
    depth = 0
    for root, folders, files in os.walk(folder):
        depth = depth + 1
        signatures = dict()
        for file in files:
            if file == NVFLARE_SIG_FILE or file == NVFLARE_SUBMITTER_CRT_FILE:
                continue
            with open(os.path.join(root, file), "rb") as f:
                signatures[file] = sign_content(
                    content=f.read(),
                    signing_pri_key=signing_pri_key,
                )
        for folder in folders:
            signatures[folder] = sign_content(
                content=folder,
                signing_pri_key=signing_pri_key,
            )

        with open(os.path.join(root, NVFLARE_SIG_FILE), "wt") as f:
            json.dump(signatures, f)
        shutil.copyfile(crt_path, os.path.join(root, NVFLARE_SUBMITTER_CRT_FILE))
        if depth >= max_depth:
            break


def verify_folder_signature(src_folder, root_ca_path):
    try:
        root_ca_cert = load_crt(root_ca_path)
        root_ca_public_key = root_ca_cert.public_key()
        for root, folders, files in os.walk(src_folder):
            try:
                with open(os.path.join(root, NVFLARE_SIG_FILE), "rt") as f:
                    signatures = json.load(f)
                cert = load_crt(os.path.join(root, NVFLARE_SUBMITTER_CRT_FILE))
                public_key = cert.public_key()
            except:
                continue  # TODO: shall return False

            verify_cert(cert_to_be_verified=cert, root_ca_public_key=root_ca_public_key)
            for file in files:
                if file == NVFLARE_SIG_FILE or file == NVFLARE_SUBMITTER_CRT_FILE:
                    continue
                signature = signatures.get(file)
                if signature:
                    with open(os.path.join(root, file), "rb") as f:
                        verify_content(
                            content=f.read(),
                            signature=signature,
                            public_key=public_key,
                        )
            for folder in folders:
                signature = signatures.get(folder)
                if signature:
                    verify_content(
                        content=folder,
                        signature=signature,
                        public_key=public_key,
                    )
        return True
    except Exception as e:
        return False


def sign_all(content_folder, signing_pri_key):
    signatures = dict()
    for f in os.listdir(content_folder):
        path = os.path.join(content_folder, f)
        if os.path.isfile(path):
            with open(path, "rb") as file:
                signatures[f] = sign_content(
                    content=file.read(),
                    signing_pri_key=signing_pri_key,
                )
    return signatures


def load_yaml(file):

    root = os.path.split(file)[0]
    yaml_data = None
    if isinstance(file, str):
        with open(file, "r") as f:
            yaml_data = yaml.safe_load(f)
    elif isinstance(file, bytes):
        yaml_data = yaml.safe_load(file)

    yaml_data = load_yaml_include(root, yaml_data)

    return yaml_data


def load_yaml_include(root, yaml_data):
    new_data = {}
    for k, v in yaml_data.items():
        if k == "include":
            if isinstance(v, str):
                includes = [v]
            elif isinstance(v, list):
                includes = v
            for item in includes:
                new_data.update(load_yaml(os.path.join(root, item)))
        elif isinstance(v, list):
            new_list = []
            for item in v:
                if isinstance(item, dict):
                    item = load_yaml_include(root, item)
                new_list.append(item)
            new_data[k] = new_list
        elif isinstance(v, dict):
            new_data[k] = load_yaml_include(root, v)
        else:
            new_data[k] = v

    return new_data


def sh_replace(src, mapping_dict):
    result = src
    for k, v in mapping_dict.items():
        result = result.replace("{~~" + k + "~~}", str(v))
    return result


def update_project_server_name_config(project_config: dict, old_server_name, server_name) -> dict:
    update_participant_server_name(project_config, old_server_name, server_name)
    return project_config


def update_participant_server_name(project_config, old_server_name, new_server_name):
    participants = project_config["participants"]
    for p in participants:
        if p["type"] == "server" and p["name"] == old_server_name:
            p["name"] = new_server_name
            break
    return project_config


def update_server_default_host(project_config, default_host):
    """Update the default_host property of the Server in the project config.
    If a client does not explicitly specify "connect_to", it will use the default_host to connect to server.
    This is mainly used for POC, where the default_host is set to localhost.

    Args:
        project_config: the project config dict
        default_host: value of the default host

    Returns: the updated project_config

    """
    participants = project_config["participants"]
    for p in participants:
        if p["type"] == "server":
            p["default_host"] = default_host
            break
    return project_config


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


def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def _write(file_full_path, content, mode, exe=False):
    mode = mode + "w"
    with open(file_full_path, mode) as f:
        f.write(content)
    if exe:
        os.chmod(file_full_path, 0o755)


def write(file_full_path, content, mode, exe=False):
    _write(file_full_path, content, mode, exe)

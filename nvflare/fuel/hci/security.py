# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import binascii
import hashlib
import os
import uuid


class IdentityKey(object):

    NAME = "common_name"
    ORG = "organization"
    ROLE = "role"


def hash_password(password):
    """Hash a password for storing.

    Args:
        password: password to hash

    Returns: hashed password

    """
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode("ascii")
    pwd_hash = hashlib.pbkdf2_hmac(hash_name="sha512", password=password.encode("utf-8"), salt=salt, iterations=100000)

    pwd_hash = binascii.hexlify(pwd_hash)
    return (salt + pwd_hash).decode("ascii")


def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user.

    Args:
        stored_password: stored password
        provided_password: password provided by user

    Returns: True if the stored password equals the provided password, otherwise False

    """
    salt = stored_password[:64]
    stored_password = stored_password[64:]
    pwd_hash = hashlib.pbkdf2_hmac(
        hash_name="sha512", password=provided_password.encode("utf-8"), salt=salt.encode("ascii"), iterations=100000
    )

    pwd_hash = binascii.hexlify(pwd_hash).decode("ascii")
    return pwd_hash == stored_password


def make_session_token():
    """Makes a new session token.

    Returns: created session token

    """
    t = uuid.uuid1()
    return str(t)


def get_identity_info(cert: dict):
    """Gets the identity information from the provided certificate.

    Args:
        cert: certificate

    Returns: if the cert is None, returning None.
             if the cert is a dictinary, returning a dictionary containing three keys, common_name, organization and role.

    """
    if cert is None:
        return None

    cn = None
    role = None
    organization = None
    for sub in cert.get("subject", ()):
        for key, value in sub:
            if key == "commonName":
                cn = value
            elif key == "unstructuredName":
                role = value
            elif key == "organizationName":
                organization = value
    return {"common_name": cn, "organization": organization, "role": role}


def get_certificate_common_name(cert: dict):
    """Gets the common name of the provided certificate.

    Args:
        cert: certificate

    Returns: common name of provided cert

    """
    if cert is None:
        return None

    for sub in cert.get("subject", ()):
        for key, value in sub:
            if key == "commonName":
                return value


def get_certificate_identity(cert: dict) -> dict:
    """Gets the identity info of the provided certificate.

    Args:
        cert: certificate

    Returns: identity info in a dict with following keys: name, org, role

    """
    if cert is None:
        return None

    result = {}

    for sub in cert.get("subject", ()):
        for key, value in sub:
            if key == "commonName":
                result[IdentityKey.NAME] = value
            elif key == "org":
                result[IdentityKey.ORG] = value
            elif key == "role":
                result[IdentityKey.ROLE] = value
    return result

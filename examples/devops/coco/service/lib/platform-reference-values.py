#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Validate five reference fields and install a complete measurement allowlist."""

import base64
import http.client
import json
import os
import re
import ssl
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

FIELDS = {
    "snp_launch_measurement": "SNP_LAUNCH_MEASUREMENT",
    "snp_min_reported_tcb_bootloader": "SNP_MIN_REPORTED_TCB_BOOTLOADER",
    "snp_min_reported_tcb_tee": "SNP_MIN_REPORTED_TCB_TEE",
    "snp_min_reported_tcb_snp": "SNP_MIN_REPORTED_TCB_SNP",
    "snp_min_reported_tcb_microcode": "SNP_MIN_REPORTED_TCB_MICROCODE",
}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def measurements(value):
    items = [value] if type(value) is str else value
    if type(items) is not list or not 1 <= len(items) <= 64:
        raise ValueError("Measurement must be a string or a nonempty list of at most 64 measurements")
    if any(type(item) is not str or not re.fullmatch(r"[0-9a-f]{96}", item) for item in items):
        raise ValueError("Every measurement must be exactly 96 lowercase hexadecimal characters")
    if len(set(items)) != len(items):
        raise ValueError("Duplicate measurements are not allowed")
    return sorted(items)


def validate_values(values):
    if type(values) is not dict or set(values) != set(FIELDS):
        raise ValueError("Expected exactly the five documented platform-reference keys")
    measurements(values["snp_launch_measurement"])
    for key in list(FIELDS)[1:]:
        if type(values[key]) is not int or not 0 <= values[key] <= 255:
            raise ValueError(f"{key} must be an integer in 0..255, not a string or boolean")
    return values


def load_values(filename):
    with Path(filename).open("rb") as stream:
        data = stream.read(8193)
    if len(data) > 8192:
        raise ValueError("Reference file exceeds 8192 bytes")
    values = json.loads(data, object_pairs_hook=unique_object)
    return validate_values(values)


def from_environment_args(args):
    if len(args) != 5:
        raise ValueError("Expected measurement configuration and four decimal TCB floors")
    measurement, *floors = args
    if not all(re.fullmatch(r"0|[1-9][0-9]{0,2}", value) for value in floors):
        raise ValueError("TCB floors must be explicit decimal uint8 integers")
    if measurement.startswith("["):
        measurement = json.loads(measurement)
    return validate_values(dict(zip(FIELDS, [measurement, *map(int, floors)])))


def update_env(values, filename):
    validate_values(values)
    path = Path(filename)
    if path.is_symlink() or not path.is_file():
        raise ValueError("platform.env must be an existing regular, non-symlink file")
    text = path.read_text()
    for key, variable in FIELDS.items():
        value = values[key]
        if key == "snp_launch_measurement" and type(value) is list:
            # Only validated hex strings are allowed, so this single-quoted
            # JSON array cannot inject shell syntax or execute substitutions.
            assignment = f"{variable}='{json.dumps(measurements(value), separators=(',', ':'))}'"
        else:
            assignment = f'{variable}="{value}"'
        text, count = re.subn(rf"^{variable}=.*$", assignment, text, flags=re.M)
        if count != 1:
            raise ValueError(f"Expected exactly one assignment for {variable}")
    fd, temporary = tempfile.mkstemp(prefix=".platform.env.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def compare_reference(values, key, raw):
    # The pinned kbs-client prints JSON whose payload may itself be encoded JSON.
    actual = json.loads(raw)
    if isinstance(actual, str):
        actual = json.loads(actual)
    expected = measurements(values[key]) if key == "snp_launch_measurement" else values[key]
    if key == "snp_launch_measurement":
        if type(actual) is not list or measurements(actual) != expected:
            raise ValueError(f"RVPS {key}: expected exactly {expected!r}, received {actual!r}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"RVPS {key}: expected {expected!r}, received {actual!r}")
    print(f"PASS {key} = {values[key]}")


def reference_message(values):
    validate_values(values)
    payload = {"snp_launch_measurement": measurements(values["snp_launch_measurement"])}
    return {
        "version": "0.1.0",
        "type": "sample",
        "payload": base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode(),
    }


def install_measurements(values, url, cert_file, token_file):
    """One authenticated POST replaces the whole list. Never follow redirects."""
    message = reference_message(values)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError("KBS URL must be an HTTPS origin without credentials, path, query or fragment")
    with Path(token_file).open() as stream:
        token_text = stream.read(16386)
    if len(token_text) > 16385:
        raise ValueError("Invalid admin token file")
    token = token_text.strip()
    if not token or len(token) > 16384 or any(c.isspace() for c in token):
        raise ValueError("Invalid admin token file")
    context = ssl.create_default_context(cafile=cert_file)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    connection = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, context=context, timeout=30)
    try:
        connection.request(
            "POST",
            "/kbs/v0/reference-value",
            body=json.dumps(message).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
        )
        response = connection.getresponse()
        if response.status != 200:
            # Do not echo response bodies: an upstream error may reflect secrets.
            raise ValueError(f"KBS reference update failed: HTTP {response.status}; no redirect or retry performed")
    finally:
        connection.close()
    print(f"Installed complete allowlist: {len(measurements(values['snp_launch_measurement']))} measurement(s)")


def main():
    if sys.argv[1:2] == ["from-env"]:
        print(json.dumps(from_environment_args(sys.argv[2:]), indent=2))
        return
    mode, filename, *args = sys.argv[1:]
    values = load_values(filename)
    if mode == "validate" and not args:
        print(json.dumps(values, indent=2))
    elif mode == "update-env" and len(args) == 1:
        update_env(values, args[0])
    elif mode == "compare-reference" and len(args) == 2 and args[0] in FIELDS:
        compare_reference(values, args[0], args[1])
    elif mode == "install-measurements" and len(args) == 3:
        install_measurements(values, *args)
    else:
        raise ValueError(
            "Usage: platform-reference-values.py validate|update-env|compare-reference|install-measurements FILE [ARGS], or from-env MEASUREMENTS BL TEE SNP MICROCODE"
        )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, http.client.HTTPException) as error:
        sys.exit(f"ERROR: {error}")

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

"""Configure and install reviewed SNP or TDX references without changing policies."""

import base64
import http.client
import importlib.util
import json
import ssl
import sys
from pathlib import Path
from urllib.parse import urlsplit

_schema_file = Path(__file__).with_name("platform-reference-schema.py")
if not _schema_file.is_file():
    _schema_file = Path(__file__).resolve().parents[2] / "shared/platform-reference-values.py"
_spec = importlib.util.spec_from_file_location("coco_platform_reference_schema", _schema_file)
_schema = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_schema)
FIELDS = _schema.FIELDS
TDX_REFERENCE_ID = _schema.TDX_REFERENCE_ID
load_values = _schema.load_values
measurements = _schema.measurements
validate_values = _schema.validate_values
reference_payload = _schema.reference_payload
unique_object = _schema.unique_object


def compare_reference(values, key, raw):
    # kbs-client may print JSON whose payload is itself encoded JSON.
    actual = json.loads(raw, object_pairs_hook=unique_object)
    if isinstance(actual, str):
        actual = json.loads(actual, object_pairs_hook=unique_object)
    expected = reference_payload(values)[key]
    if key == "snp_launch_measurement":
        if type(actual) is not list or measurements(actual) != expected:
            raise ValueError(f"RVPS {key}: expected exactly {expected!r}, received {actual!r}")
    elif key == TDX_REFERENCE_ID:
        received = {"schema": _schema.TDX_SCHEMA, "tee": "tdx", "profiles": actual}
        if reference_payload(received)[key] != expected:
            raise ValueError("RVPS TDX profiles differ from the complete approved profile set")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"RVPS {key}: expected {expected!r}, received {actual!r}")
    print(f"PASS {key} = {expected}")


def reference_message(values):
    validate_values(values)
    payload = reference_payload(values)
    if "snp_launch_measurement" in payload:
        payload = {"snp_launch_measurement": payload["snp_launch_measurement"]}
    return message_for_payload(payload)


def message_for_payload(payload):
    return {
        "version": "0.1.0",
        "type": "sample",
        "payload": base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode(),
    }


def install_measurements(values, url, cert_file, token_file):
    """One authenticated POST replaces the whole list. Never follow redirects."""
    post_reference(reference_message(values), url, cert_file, token_file)


def install_references(values, url, cert_file, token_file):
    payload = reference_payload(values)
    if values.get("tee") == "tdx":
        # One key/value replacement: no intermediate mixed profile state.
        post_reference(message_for_payload(payload), url, cert_file, token_file)
    else:
        # A floor of 255 is not an unconditional deny. Clear measurements before
        # replacing floors, and activate the allowlist only after floors succeed.
        post_reference(message_for_payload({"snp_launch_measurement": []}), url, cert_file, token_file)
        floors = {key: value for key, value in payload.items() if key != "snp_launch_measurement"}
        post_reference(message_for_payload(floors), url, cert_file, token_file)
        install_measurements(values, url, cert_file, token_file)
    print("Installed complete reference set; other TEE references are unchanged")


def post_reference(message, url, cert_file, token_file):
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


def main():
    mode, filename, *args = sys.argv[1:]
    values = load_values(filename)
    if mode == "validate" and not args:
        print(json.dumps(values, indent=2))
    elif mode == "reference-ids" and not args:
        print("\n".join(reference_payload(values)))
    elif mode == "compare-reference" and len(args) == 2 and args[0] in reference_payload(values):
        compare_reference(values, args[0], args[1])
    elif mode == "install-measurements" and len(args) == 3:
        install_measurements(values, *args)
    elif mode == "install-references" and len(args) == 3:
        install_references(values, *args)
    else:
        raise ValueError(
            "Usage: platform-reference-values.py validate|reference-ids|compare-reference|install-references FILE [ARGS]"
        )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, http.client.HTTPException) as error:
        sys.exit(f"ERROR: {error}")

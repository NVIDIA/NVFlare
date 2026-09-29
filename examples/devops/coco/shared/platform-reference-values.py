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

"""Strict reference handoff schema shared by collection and approval tools.

TDX fields use the pinned Trustee verifier's lowercase, byte-order-preserving
hex encoding (including the eight XFAM bytes), not human-readable integers.
One RVPS value contains complete profiles to prevent cross-profile mixing or
transient combinations during reference replacement.
"""

import json
import re
from pathlib import Path

FIELDS = {
    "snp_launch_measurement": "SNP_LAUNCH_MEASUREMENT",
    "snp_min_reported_tcb_bootloader": "SNP_MIN_REPORTED_TCB_BOOTLOADER",
    "snp_min_reported_tcb_tee": "SNP_MIN_REPORTED_TCB_TEE",
    "snp_min_reported_tcb_snp": "SNP_MIN_REPORTED_TCB_SNP",
    "snp_min_reported_tcb_microcode": "SNP_MIN_REPORTED_TCB_MICROCODE",
}
TDX_FIELDS = ("mr_td", "rtmr_1", "rtmr_2", "xfam", "tdvfkernel", "tdvfkernelparams")
TDX_REFERENCE_ID = "coco_tdx_profiles_v2"
TDX_SCHEMA = "coco-platform-reference-values/v2"
MAX_REFERENCE_BYTES = 131072


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
    if type(values) is not dict:
        raise ValueError("Expected a platform-reference JSON object")
    if set(values) == set(FIELDS):
        measurements(values["snp_launch_measurement"])
        for key in list(FIELDS)[1:]:
            if type(values[key]) is not int or not 0 <= values[key] <= 255:
                raise ValueError(f"{key} must be an integer in 0..255, not a string or boolean")
        return values
    if set(values) != {"schema", "tee", "profiles"} or values["schema"] != TDX_SCHEMA or values["tee"] != "tdx":
        raise ValueError("Expected the five SNP keys or the documented v2 TDX schema")
    profiles = values["profiles"]
    if type(profiles) is not list or not 1 <= len(profiles) <= 64:
        raise ValueError("TDX profiles must be a nonempty list of at most 64 complete profiles")
    identifiers, tuples = set(), set()
    for profile in profiles:
        if type(profile) is not dict or set(profile) != {"id", *TDX_FIELDS}:
            raise ValueError("Each TDX profile must contain exactly id and all six measurement fields")
        identifier = profile["id"]
        if type(identifier) is not str or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", identifier):
            raise ValueError(
                "TDX profile id must be 1..64 lowercase alphanumeric, dot, underscore or hyphen characters"
            )
        for field in TDX_FIELDS:
            width = 16 if field == "xfam" else 96
            if type(profile[field]) is not str or not re.fullmatch(rf"[0-9a-f]{{{width}}}", profile[field]):
                raise ValueError(f"TDX {field} must be exactly {width} lowercase hexadecimal characters")
        measurement_tuple = tuple(profile[field] for field in TDX_FIELDS)
        if identifier in identifiers or measurement_tuple in tuples:
            raise ValueError("Duplicate TDX profile ids or measurement tuples are not allowed")
        identifiers.add(identifier)
        tuples.add(measurement_tuple)
    return values


def load_values(filename):
    with Path(filename).open("rb") as stream:
        data = stream.read(MAX_REFERENCE_BYTES + 1)
    if len(data) > MAX_REFERENCE_BYTES:
        raise ValueError(f"Reference file exceeds {MAX_REFERENCE_BYTES} bytes")
    return validate_values(json.loads(data, object_pairs_hook=unique_object))


def reference_payload(values):
    """Return the complete expected RVPS values (no evidence-selected paths)."""
    validate_values(values)
    if values.get("tee") == "tdx":
        return {TDX_REFERENCE_ID: sorted(values["profiles"], key=lambda profile: profile["id"])}
    return {**values, "snp_launch_measurement": measurements(values["snp_launch_measurement"])}

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

"""Validate independently approved GPU versions and update their RVPS references."""

import importlib.util
import json
import os
import re
import shlex
import sys
import tempfile
from pathlib import Path

SCHEMA = "coco-gpu-reference-values/v1"
IDS = ("allowed_driver_versions", "allowed_vbios_versions")
MAX_BYTES = 16384
VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}\Z")

_spec = importlib.util.spec_from_file_location(
    "platform_reference_transport", Path(__file__).with_name("platform-reference-values.py")
)
_transport = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_transport)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate GPU reference key: " + key)
        result[key] = value
    return result


def validate_values(values):
    if type(values) is not dict or set(values) != {"schema", *IDS} or values["schema"] != SCHEMA:
        raise ValueError("Expected the complete versioned GPU reference schema")
    for name in IDS:
        versions = values[name]
        if (
            type(versions) is not list
            or not 1 <= len(versions) <= 64
            or any(type(version) is not str or not VERSION.fullmatch(version) for version in versions)
            or len(set(versions)) != len(versions)
        ):
            raise ValueError("GPU reference must contain 1..64 unique exact version strings: " + name)
    return values


def load_values(filename):
    with Path(filename).open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("GPU reference file exceeds 16384 bytes")
    return validate_values(json.loads(data, object_pairs_hook=unique_object))


def payload(values):
    validate_values(values)
    return {name: sorted(values[name]) for name in IDS}


def stage_payload(values, stage):
    approved = payload(values)
    if stage == "clear":
        return {IDS[0]: []}
    if stage == "vbios":
        return {IDS[1]: approved[IDS[1]]}
    if stage == "drivers":
        return {IDS[0]: approved[IDS[0]]}
    raise ValueError("Unknown GPU reference stage")


def expected_stage(values, stage):
    return next(iter(stage_payload(values, stage).items()))


def compare_stage(values, stage, raw):
    name, expected = expected_stage(values, stage)
    actual = json.loads(raw, object_pairs_hook=unique_object)
    if type(actual) is str:
        actual = json.loads(actual, object_pairs_hook=unique_object)
    if type(actual) is not list or actual != expected:
        raise ValueError(f"RVPS {name} differs from the approved {stage} stage")


def post_stage(values, stage, url, cert_file, token_file):
    message = _transport.message_for_payload(stage_payload(values, stage))
    _transport.post_reference(message, url, cert_file, token_file)


def update_env(values, filename):
    validate_values(values)
    path = Path(filename)
    if path.is_symlink() or not path.is_file():
        raise ValueError("platform.env must be an existing regular, non-symlink file")
    snapshot = path.parent / "approved-gpu-reference-values.json"
    if snapshot.is_symlink():
        raise ValueError("GPU reference snapshot must not be a symlink")
    text = path.read_text()
    assignment = "GPU_REFERENCE_VALUES_FILE=" + shlex.quote(str(snapshot.resolve()))
    text, count = re.subn(r"^GPU_REFERENCE_VALUES_FILE=.*$", lambda _: assignment, text, flags=re.M)
    if count > 1:
        raise ValueError("Expected at most one GPU_REFERENCE_VALUES_FILE assignment")
    if count == 0:
        text += ("" if text.endswith("\n") or not text else "\n") + assignment + "\n"
    serialized = json.dumps(values, separators=(",", ":"))
    if len(serialized.encode()) > MAX_BYTES:
        raise ValueError("GPU reference snapshot exceeds 16384 bytes")
    atomic_write(snapshot, serialized)
    atomic_write(path, text)


def atomic_write(path, content):
    fd, temporary = tempfile.mkstemp(prefix=".gpu-reference.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        raise ValueError("Usage: gpu-reference-values.py MODE VALUES.json [ARGS]")
    mode, filename, *args = argv
    values = load_values(filename)
    if mode == "validate" and not args:
        return
    if mode == "update-env" and len(args) == 1:
        update_env(values, args[0])
        return
    if mode == "reference-id" and len(args) == 1:
        print(expected_stage(values, args[0])[0])
        return
    if mode == "compare-stage" and len(args) == 2:
        compare_stage(values, args[0], args[1])
        return
    if mode == "post-stage" and len(args) == 4:
        post_stage(values, args[0], *args[1:])
        return
    raise ValueError("Invalid GPU reference operation")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        sys.exit(f"ERROR: {error}")

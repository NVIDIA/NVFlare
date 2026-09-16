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

"""Derive/check the single reviewed NVFlare override without changing other Kata settings."""

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path

OPTION = "agent.guest_components_rest_api"
REQUIRED = OPTION + "=all"
CAPABILITY = "guest-local-aa-token/v1"


def require_token_api(cmdline):
    if not isinstance(cmdline, str):
        raise ValueError("Missing kernel command line")
    values = [part for part in cmdline.split() if part == OPTION or part.startswith(OPTION + "=")]
    if values != [REQUIRED]:
        raise ValueError("Require exactly one " + REQUIRED)


def read_config(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular Kata configuration file")
    return path.read_bytes().decode("utf-8")


def runtime_config_path(path):
    """Follow Kata's runtime link without allowing a write outside its config tree."""
    root = path.absolute().parent.resolve(strict=True)
    target = path.resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("Installed Kata configuration must be a regular file within its configuration directory")
    return target


def settings(text):
    # Ignore comments/layout, not value types (Python equality considers True == 1).
    return json.dumps(tomllib.loads(text), sort_keys=True, allow_nan=False)


def derive(text):
    original = tomllib.loads(text)
    params = original["hypervisor"]["qemu"]["kernel_params"]
    if not isinstance(params, str):
        raise ValueError("Expected explicit string kernel_params")
    tokens = [part for part in params.split() if part == OPTION or part.startswith(OPTION + "=")]
    if len(tokens) > 1 or (tokens and tokens[0] not in [OPTION + "=" + v for v in ("all", "resource", "attestation")]):
        raise ValueError("Conflicting or unsupported guest-components API setting")
    if tokens == [REQUIRED]:
        return text
    # Preserve whitespace and repeated unrelated options such as pci=; never use a key/value dict.
    updated = re.sub(r"(?<!\S)" + re.escape(OPTION) + r"=\S+", REQUIRED, params) if tokens else params + " " + REQUIRED
    section = re.search(r"(?ms)^\[hypervisor\.qemu\][ \t]*(?:#[^\n]*)?\n(?P<body>.*?)(?=^\[|\Z)", text)
    if not section:
        raise ValueError("Unsupported hypervisor.qemu TOML layout")
    body = section.group("body")
    field = re.compile(r"""(?m)^(kernel_params\s*=\s*)("(?:[^"\\\n]|\\.)*"|'[^'\n]*')([ \t]*(?:#[^\n]*)?)$""")
    if len(list(field.finditer(body))) != 1:
        raise ValueError("Expected one single-line kernel_params assignment")
    body = field.sub(lambda m: m[1] + json.dumps(updated) + m[3], body)
    result = text[: section.start("body")] + body + text[section.end("body") :]
    expected = copy.deepcopy(original)
    expected["hypervisor"]["qemu"]["kernel_params"] = updated
    if tomllib.loads(result) != expected:
        raise ValueError("Runtime derivation changed an unrelated setting")
    require_token_api(updated)
    return result


def verify(upstream, approved, record, installed=None, launch=None):
    raw = read_config(upstream)
    effective = read_config(approved)
    if effective != derive(raw):
        raise ValueError("Approved configuration differs from the reviewed upstream derivation")
    expected = {
        "schema": "coco-kata-runtime-profile/v1",
        "capability": CAPABILITY,
        "upstream_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "approved_sha256": hashlib.sha256(effective.encode()).hexdigest(),
    }
    if json.loads(record.read_text()) != expected:
        raise ValueError("Runtime configuration provenance changed")
    installed_text = None
    if installed:
        installed_text = read_config(runtime_config_path(installed))
        if settings(installed_text) != settings(effective):
            raise ValueError("Installed Kata configuration differs from the approved derivation")
    if launch:
        if installed_text is None:
            raise ValueError("Launch verification requires the installed configuration")
        captured = json.loads(launch.read_text())
        require_token_api(captured["launch_inputs"]["kernel_command_line"])
        if captured["artifacts"]["kata_config"]["sha256"] != hashlib.sha256(installed_text.encode()).hexdigest():
            raise ValueError("Captured Kata configuration differs from the verified installed file")


def install(upstream, approved, record, installed):
    verify(upstream, approved, record)
    target = runtime_config_path(installed)
    current = settings(read_config(target))
    if current not in (settings(read_config(upstream)), settings(read_config(approved))):
        raise ValueError("Installed Kata settings differ from both upstream and approved profile")
    enable(target)
    verify(upstream, approved, record, installed)


def enable(path):
    path = runtime_config_path(path)
    original = read_config(path)
    updated = derive(original)
    if updated == original:
        return
    info = path.stat()
    # Atomic replacement: a failed write must not truncate a runtime configuration.
    fd, temp = tempfile.mkstemp(prefix=".nvflare-kata-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchown(stream.fileno(), info.st_uid, info.st_gid)
            os.fchmod(stream.fileno(), info.st_mode & 0o777)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for name in ("derive", "verify", "install"):
        p = sub.add_parser(name)
        p.add_argument("upstream", type=Path)
        p.add_argument("approved", type=Path)
        p.add_argument("record", type=Path)
        if name in ("verify", "install"):
            p.add_argument("--installed", type=Path, required=name == "install")
        if name == "verify":
            p.add_argument("--launch", type=Path)
    for name in ("enable", "check"):
        p = sub.add_parser(name)
        p.add_argument("config", type=Path)
        if name == "check":
            p.add_argument("--runtime", type=Path, help="also verify kata-env effective parameters")
    args = parser.parse_args()
    if args.operation == "derive":
        raw = read_config(args.upstream)
        effective = derive(raw)
        if any(p.exists() or p.is_symlink() for p in (args.approved, args.record)):
            raise ValueError("Preserve existing evidence; choose a new profile")
        with args.approved.open("x") as stream:
            stream.write(effective)
        with args.record.open("x") as stream:
            json.dump(
                {
                    "schema": "coco-kata-runtime-profile/v1",
                    "capability": CAPABILITY,
                    "upstream_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "approved_sha256": hashlib.sha256(effective.encode()).hexdigest(),
                },
                stream,
                indent=2,
            )
            stream.write("\n")
        verify(args.upstream, args.approved, args.record)
    elif args.operation == "verify":
        verify(args.upstream, args.approved, args.record, args.installed, args.launch)
    elif args.operation == "install":
        install(args.upstream, args.approved, args.record, args.installed)
    elif args.operation == "enable":
        enable(args.config)
    else:
        require_token_api(
            tomllib.loads(read_config(runtime_config_path(args.config)))["hypervisor"]["qemu"]["kernel_params"]
        )
        if args.runtime:
            env = json.loads(
                subprocess.check_output(
                    [str(args.runtime), "--config", str(args.config), "kata-env", "--json"], text=True, timeout=30
                )
            )
            require_token_api(env["Kernel"]["Parameters"])
    print("Verified NVFlare guest-local token API runtime setting")


if __name__ == "__main__":
    main()

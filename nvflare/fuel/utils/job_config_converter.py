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

"""Render HOCON and OmegaConf job configs on the author's machine.

FLARE sites parse .conf files as JSON and never resolve YAML interpolations or aliases.
Includes, substitutions, and interpolations are resolved here with the author's files and environment.
"""

import glob
import json
import os
import shutil

from nvflare.fuel_opt.utils.omegaconf_loader import reject_yaml_aliases


def _needs_rendering(path: str) -> bool:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        if path.endswith(".conf"):
            json.loads(text)
        else:
            reject_yaml_aliases(text)
    except Exception:
        return True
    return path.endswith((".yml", ".yaml")) and "${" in text


def _render(path: str) -> str:
    if path.endswith(".conf"):
        from pyhocon import ConfigFactory, HOCONConverter

        return HOCONConverter.to_json(ConfigFactory.parse_file(path))

    from omegaconf import OmegaConf

    return OmegaConf.to_yaml(OmegaConf.to_container(OmegaConf.load(path), resolve=True))


def convert_job_configs(job_dir: str, work_dir: str) -> str:
    """Returns job_dir, or a copy under work_dir with HOCON and OmegaConf configs rendered as data."""
    job_dir = os.path.abspath(job_dir)
    root = glob.escape(job_dir)
    # Legacy app folders without meta keep their configs in config/ directly.
    patterns = ["meta.*", os.path.join("config", "*"), os.path.join("*", "config", "*")]
    rel_paths = [
        os.path.relpath(path, job_dir)
        for pattern in patterns
        for path in glob.glob(os.path.join(root, pattern))
        if path.endswith((".conf", ".yml", ".yaml")) and _needs_rendering(path)
    ]
    if not rel_paths:
        return job_dir

    converted_dir = os.path.join(os.path.abspath(work_dir), os.path.basename(job_dir))
    shutil.rmtree(converted_dir, ignore_errors=True)
    shutil.copytree(job_dir, converted_dir, symlinks=True)
    for rel_path in rel_paths:
        try:
            text = _render(os.path.join(job_dir, rel_path))
        except Exception as e:
            raise ValueError(f"cannot render job config {rel_path}: {e}") from e
        with open(os.path.join(converted_dir, rel_path), "w", encoding="utf-8") as f:
            f.write(text)
    return converted_dir

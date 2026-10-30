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

Includes, substitutions, and interpolations can read site files, URLs, and environment variables,
so the server rejects submitted job configs that still use them. Job tools resolve them here
with the author's files and environment.
"""

import glob
import json
import os
import shutil

from nvflare.fuel.utils.config import ConfigFormat


def load_omegaconf_job_config(path: str) -> dict:
    """Loads a YAML job config with OmegaConf aliases and interpolations resolved, for author-side tools."""
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def used_configs(paths):
    """Keeps, per folder and config name, the file runtime lookup picks first by extension precedence."""
    ext_formats = ConfigFormat.config_ext_formats()
    exts = list(ext_formats)
    used = {}
    for path in paths:
        ext = next((ext for ext in exts if path.endswith(ext)), None)
        if ext is None:
            continue
        key = path[: -len(ext)]
        if key not in used or exts.index(ext) < exts.index(used[key][1]):
            used[key] = (path, ext)
    return [(path, ext_formats[ext]) for path, ext in used.values()]


def needs_rendering(text: str, fmt: ConfigFormat) -> bool:
    """Whether a HOCON or OmegaConf config uses features beyond plain data, or YAML aliases."""
    try:
        if fmt == ConfigFormat.PYHOCON:
            json.loads(text)
            return False
        import yaml

        # Aliases let a few bytes expand into an exponentially large config.
        return "${" in text or any(isinstance(e, yaml.AliasEvent) for e in yaml.parse(text, Loader=yaml.SafeLoader))
    except Exception:
        return True


def _render(path: str, fmt: ConfigFormat) -> str:
    if fmt == ConfigFormat.PYHOCON:
        from pyhocon import ConfigFactory, HOCONConverter

        return HOCONConverter.to_json(ConfigFactory.parse_file(path))

    from omegaconf import OmegaConf

    return OmegaConf.to_yaml(load_omegaconf_job_config(path))


def convert_job_configs(job_dir: str, work_dir: str) -> str:
    """Returns job_dir, or a copy under work_dir with HOCON and OmegaConf configs rendered as data."""
    job_dir = os.path.abspath(job_dir)
    root = glob.escape(job_dir)
    # Legacy app folders without meta keep their configs in config/ directly.
    patterns = ["meta.*", os.path.join("config", "*"), os.path.join("*", "config", "*")]
    to_render = []
    for path, fmt in used_configs(path for pattern in patterns for path in glob.glob(os.path.join(root, pattern))):
        if fmt not in (ConfigFormat.PYHOCON, ConfigFormat.OMEGACONF):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            if not needs_rendering(f.read(), fmt):
                continue
        rel_path = os.path.relpath(path, job_dir)
        # Writing through a linked file or folder would overwrite a file outside the job.
        if os.path.realpath(path) != os.path.join(os.path.realpath(job_dir), rel_path):
            raise ValueError(f"cannot render linked job config {rel_path}")
        to_render.append((rel_path, fmt))
    if not to_render:
        return job_dir

    converted_dir = os.path.join(os.path.abspath(work_dir), os.path.basename(job_dir))
    shutil.rmtree(converted_dir, ignore_errors=True)
    shutil.copytree(job_dir, converted_dir, symlinks=True)
    for rel_path, fmt in to_render:
        try:
            text = _render(os.path.join(job_dir, rel_path), fmt)
        except Exception as e:
            raise ValueError(f"cannot render job config {rel_path}: {e}") from e
        with open(os.path.join(converted_dir, rel_path), "w", encoding="utf-8") as f:
            f.write(text)
    return converted_dir

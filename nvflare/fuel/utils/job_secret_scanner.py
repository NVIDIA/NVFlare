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

"""Secret validation for generated NVFlare job configuration files."""

import json
import os
from typing import List, Optional

from nvflare.fuel.utils.secret_utils import SecretFinding, warn_on_potential_secrets

_GENERATED_JOB_CONFIG_FILES = frozenset(
    {
        "config_fed_client.json",
        "config_fed_server.json",
        "meta.json",
    }
)


def warn_on_potential_secrets_in_job_dir(
    job_dir: str,
    job_name: Optional[str] = None,
    context: str = "generated job file",
) -> List[SecretFinding]:
    """Warn about potential secrets in generated job JSON files.

    This is a last-line, best-effort scan of the artifact that will be exported,
    simulated, or submitted. User code under ``custom/`` is intentionally excluded.

    Args:
        job_dir: Export root or the generated job directory itself.
        job_name: Optional job name when ``job_dir`` is an export root.
        context: Description used to identify a scanned file in warning messages.
    Returns:
        All potential-secret findings emitted while scanning the job.
    """
    job_root = os.path.join(job_dir, job_name) if job_name else job_dir
    if not os.path.isdir(job_root):
        job_root = job_dir

    findings = []
    for root, dirs, files in os.walk(job_root):
        dirs[:] = [directory for directory in dirs if directory != "custom"]
        for file_name in files:
            if file_name not in _GENERATED_JOB_CONFIG_FILES:
                continue

            path = os.path.join(root, file_name)
            try:
                with open(path, encoding="utf-8") as file:
                    data = json.load(file)
            except (OSError, UnicodeError, ValueError):
                # Job generation owns the validity of these files. Secret scanning is
                # best-effort and must not mask a more useful error from that layer.
                continue

            # Only use the allow-listed basename in warning context. Job and app directory names
            # are user-controlled and could themselves contain credential material.
            findings.extend(warn_on_potential_secrets(data, context=f"{context} '{file_name}'"))

    return findings

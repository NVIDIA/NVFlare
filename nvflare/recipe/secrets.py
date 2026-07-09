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

"""Public helpers for keeping secrets out of recipe parameters.

Recipe parameters (``train_args``, ``task_args``, ``eval_args``, ``per_site_config``,
``add_client_config`` / ``add_server_config`` dicts, exec params, and other nested recipe
values) can be written in clear text into generated job configuration. They must never
contain actual secret values.

Instead, either read secrets from environment variables or mounted secret files inside your
training code, or put a :func:`secret_ref` or :func:`secret_file_ref` placeholder in a supported
runtime parameter::

    from nvflare.recipe.secrets import secret_file_ref, secret_ref

    recipe = FedAvgRecipe(
        ...,
        train_args=f"--epochs 5 --api-key {secret_ref('MY_API_KEY')}",
    )
    recipe.add_client_config(
        {"service_password": secret_file_ref("/var/run/secrets/service/password")}
    )

Only the placeholders appear in the exported job. Secret references are supported at two explicit
runtime boundaries:

* command arguments consumed by NVFlare's task script runner or subprocess launcher (including
  recipe ``train_args``, ``task_args``, and ``eval_args`` that use those runners); these resolve
  after argument tokenization, immediately before the script or process starts;
* values explicitly read from a runtime job JSON file with
  :func:`nvflare.utils.configs.get_job_config_value` or its
  :func:`nvflare.utils.configs.get_client_config_value` and
  :func:`nvflare.utils.configs.get_server_config_value` wrappers. A typical recipe adds a
  top-level value with ``add_client_config`` or ``add_server_config``; nested strings resolve
  when that value is read.

Environment references are read from the executing process's environment; file references are
read from that site's filesystem. Resolved values stay in runtime memory and are not written back
to generated configuration. Dictionary keys are not resolved. Arbitrary component constructor
arguments, metadata, custom files, and other job artifacts do not resolve references. Read secrets
inside user code for those cases. Mounted-file reference paths cannot contain whitespace or braces.

Recipes automatically scan their parameters and emit :class:`PotentialSecretWarning` when a
value looks like an actual secret. Detection is best-effort and does not make a supplied value
safe. A valid reference in an explicitly unsupported parameter emits
:class:`UnsupportedSecretRefWarning`. See
:func:`nvflare.fuel.utils.secret_utils.find_potential_secrets` for the heuristics.
"""

from nvflare.fuel.utils.secret_utils import (
    SECRET_REF_PATTERN,
    PotentialSecretWarning,
    SecretFinding,
    UnsupportedSecretRefWarning,
    find_potential_secrets,
    has_secret_refs,
    resolve_secret_refs,
    secret_file_ref,
    secret_ref,
    warn_on_potential_secrets,
    warn_on_unsupported_secret_ref_keys,
    warn_on_unsupported_secret_refs,
    warn_on_unsupported_secret_refs_outside_keys,
)

__all__ = [
    "PotentialSecretWarning",
    "UnsupportedSecretRefWarning",
    "SecretFinding",
    "SECRET_REF_PATTERN",
    "secret_ref",
    "secret_file_ref",
    "has_secret_refs",
    "resolve_secret_refs",
    "find_potential_secrets",
    "warn_on_potential_secrets",
    "warn_on_unsupported_secret_refs",
    "warn_on_unsupported_secret_ref_keys",
    "warn_on_unsupported_secret_refs_outside_keys",
]

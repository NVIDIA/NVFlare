# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, List, Optional, Union

# Single source of truth for the default export dir. Kept in the current working
# directory on purpose: an export is a user-requested artifact, so it lands next to
# where the script ran (discoverable and predictable), and a relative path stays
# portable across OSes -- unlike a fixed /tmp path, which is non-portable on Windows
# and collision-prone on shared hosts.
DEFAULT_EXPORT_DIR = "./fl_job"

_CONSUMED = False


def _consume_recipe_args() -> tuple:
    """Strip --export / --export-dir from sys.argv and return (export, export_dir).

    Called once at module import time so that the caller's argparse never sees
    these flags regardless of the order in which parse_args() and execute() appear
    in job.py.

    Transactional: sys.argv is only mutated if the parse is clean. A malformed
    (dangling) --export-dir aborts the pass without mutating sys.argv and without
    enabling export, so a malformed import can neither raise nor silently export.
    The decision is frozen after the first call so repeated direct calls return the
    recorded import-time result rather than re-scanning a since-mutated sys.argv.
    """
    global _CONSUMED
    if _CONSUMED:
        return _RECIPE_EXPORT, _RECIPE_EXPORT_DIR

    argv = sys.argv[1:]
    export = False
    export_dir = DEFAULT_EXPORT_DIR
    remaining = []
    i = 0
    while i < len(argv):
        if argv[i] == "--export":
            export = True
            i += 1
        elif argv[i] == "--export-dir":
            if i + 1 >= len(argv):
                # Dangling --export-dir with no value: abort the entire pass. Do not
                # mutate sys.argv and do not enable export. Leaving argv intact lets the
                # caller's own parser surface the leftover flags; enabling export here
                # could export to the default dir against the user's intent (e.g. under
                # parse_known_args()). Freeze the decision so a later direct call returns
                # it instead of re-scanning a since-mutated sys.argv.
                _CONSUMED = True
                return False, DEFAULT_EXPORT_DIR
            export_dir = argv[i + 1]
            i += 2
        elif argv[i].startswith("--export-dir="):
            export_dir = argv[i].split("=", 1)[1]
            i += 1
        else:
            remaining.append(argv[i])
            i += 1
    sys.argv[1:] = remaining
    _CONSUMED = True
    return export, export_dir


# Intentional import-time sys.argv mutation: strip --export / --export-dir before
# any ArgumentParser.parse_args() call in job.py runs. Doing this lazily (e.g. inside
# execute()) would be too late if the caller calls parse_args() first, which is the
# common pattern. The mutation is safe because job.py is always the process entry point.
_RECIPE_EXPORT, _RECIPE_EXPORT_DIR = _consume_recipe_args()


def _peek_recipe_args() -> tuple:
    """Return the export flags consumed at import time."""
    return _RECIPE_EXPORT, _RECIPE_EXPORT_DIR


from nvflare.apis.filter import Filter
from nvflare.app_common.widgets.decomposer_reg import DecomposerRegister
from nvflare.fuel.utils.fobs import Decomposer
from nvflare.fuel.utils.job_secret_scanner import warn_on_potential_secrets_in_job_dir
from nvflare.fuel.utils.secret_utils import (
    warn_on_potential_secrets,
    warn_on_unsupported_secret_ref_keys,
    warn_on_unsupported_secret_refs,
    warn_on_unsupported_secret_refs_outside_keys,
)
from nvflare.job_config.api import FedJob
from nvflare.job_config.defs import FilterType


class ExecEnv(ABC):

    def __init__(self, extra: Optional[dict] = None):
        """Constructor of ExecEnv

        Args:
            extra: a dict of extra properties
        """
        if extra is None:
            extra = {}
        if not isinstance(extra, dict):
            raise ValueError(f"extra must be dict but got {type(extra)}")
        self.extra = extra

    def get_extra_prop(self, prop_name: str, default=None):
        """Get the specified extra property.

        Args:
            prop_name: name of the property
            default: the default value to return if the named property does not exist.

        Returns: value of the property or the default

        """
        return self.extra.get(prop_name, default)

    @abstractmethod
    def deploy(self, job: FedJob) -> str:
        """Deploy a FedJob and return an execution response.

        Args:
            job: The FedJob to deploy.

        Returns:
            str: The job ID.
        """
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> Optional[str]:
        """Get the status of a job.

        Args:
            job_id: The job ID to check status for.

        Returns:
            Optional[str]: The status of the job, or None if not supported.
        """
        pass

    @abstractmethod
    def abort_job(self, job_id: str) -> None:
        """Abort a running job.

        Args:
            job_id: The job ID to abort.
        """
        pass

    @abstractmethod
    def get_job_result(self, job_id: str, timeout: float = 0.0) -> Optional[str]:
        """Get the result workspace of a job.

        Args:
            job_id: The job ID to get results for.
            timeout: The timeout for the job to complete. Defaults to 0.0 (no timeout).

        Returns:
            Optional[str]: The result workspace path if job completed, None if still running or stopped early.
        """
        pass

    def stop(self, clean_up: bool = False) -> None:
        """Stop the execution environment and optionally clean up resources.

        This method is called after job execution to ensure proper cleanup.
        Default implementation is a no-op. Override in subclasses that need cleanup.

        Args:
            clean_up: If True, remove workspace and temporary files after stopping.
                      If False, only stop running processes but preserve workspace.
                      Defaults to False.
        """
        pass


class Recipe(ABC):

    _SECRET_PARAMETER_ATTRS = (
        "train_args",
        "eval_args",
        "task_args",
        "command",
        "task_data",
        "task_meta",
        "per_site_config",
        "server_config_overrides",
        "client_config_overrides",
        "optimizer_args",
        "lr_scheduler_args",
        "xgb_params",
        "statistic_configs",
        "extra_env",
        "run_config",
        "device_training_params",
    )

    _UNSUPPORTED_SECRET_REF_ATTRS = frozenset(
        {
            "task_data",
            "task_meta",
            "server_config_overrides",
            "optimizer_args",
            "lr_scheduler_args",
            "xgb_params",
            "statistic_configs",
            "extra_env",
            "run_config",
            "device_training_params",
            "per_site_config",
        }
    )

    _SUPPORTED_PER_SITE_SECRET_REF_KEYS = None

    def __init__(self, job: FedJob):
        """This is base class of a recipe. Recipes are implemented by jobs.
        A concrete recipe must provide the job for recipe implementation.

        Security contract -- no secrets in recipe parameters:
            Recipe parameters (``train_args``, ``task_args``, ``eval_args``,
            ``per_site_config``, config overrides, dicts passed to
            ``add_client_config``/``add_server_config``, exec params, etc.) can be written in
            clear text into generated job configuration. These parameters and their nested
            values must never contain actual passwords, API keys, tokens, private keys, or
            other credentials. Instead, read secrets from site environment variables or
            mounted secret files inside your code, or pass a placeholder created with
            :func:`nvflare.recipe.secrets.secret_ref` or
            :func:`nvflare.recipe.secrets.secret_file_ref` at a supported runtime boundary.
            See :mod:`nvflare.recipe.secrets` for the supported parameter locations.

            Before export or run, recipes scan their parameters with heuristics and emit
            :class:`nvflare.recipe.secrets.PotentialSecretWarning` when a value looks like an
            actual secret. The scan is best-effort: absence of a warning does not prove a
            parameter is safe to share.

        Args:
            job: the job that implements the recipe.
        """
        self._job = job
        self.name = job.name
        self._configured_site_names = None
        self._client_apps_prepared = False
        self._tensor_streaming_added = False
        self._cse_added = False
        warn_on_potential_secrets(getattr(job, "name", None), context="recipe parameter 'name'")

    def _warn_potential_secrets_in_params(self):
        """Warn if common recipe parameters look like they contain actual secret values.

        Runs immediately before export or run on conventional parameter attributes that
        concrete recipes retain. Recipes that transform a parameter before calling this
        constructor validate that original value explicitly.
        """
        for attr in self._SECRET_PARAMETER_ATTRS:
            value = getattr(self, attr, None)
            if value is not None:
                context = f"recipe parameter '{attr}'"
                warn_on_potential_secrets(value, context=context)
                if attr in self._UNSUPPORTED_SECRET_REF_ATTRS:
                    if attr == "per_site_config" and self._SUPPORTED_PER_SITE_SECRET_REF_KEYS is not None:
                        warn_on_unsupported_secret_refs_outside_keys(
                            value,
                            supported_value_keys=self._SUPPORTED_PER_SITE_SECRET_REF_KEYS,
                            supported_value_depth=2,
                            context=context,
                        )
                    else:
                        warn_on_unsupported_secret_refs(value, context=context)

    def process_env(self, env: ExecEnv):
        """Process environment-specific configuration.

        Subclasses can override to add environment-specific processing.
        Script validation is handled by each ExecEnv subclass in deploy().
        """
        pass

    def set_per_site_config(self, config: Dict[str, Dict]) -> None:
        """Set helper-provided per-site configuration for this recipe.

        The generic helper validates only the site-keyed shape. Recipes that
        need to map fields into generated app config, command arguments, data
        loaders, or validators should override ``_apply_per_site_config``.
        Client topology is prepared later, before the first client-targeted
        customization or before export or execution.

        Per-site config values end up in the generated job configuration in clear
        text and must never contain actual secret values; see the Recipe class
        docstring for the recommended alternatives.
        """
        if not isinstance(config, dict):
            raise TypeError(f"per-site config must be a dict, got {type(config).__name__}")
        if not config:
            raise ValueError("per-site config must not be empty")
        for site_name, site_config in config.items():
            if not isinstance(site_name, str):
                raise TypeError(f"per-site config key must be a str, got {type(site_name).__name__}")
            if not isinstance(site_config, dict):
                raise TypeError(
                    f"per-site config for site {site_name!r} must be a dict, got {type(site_config).__name__}"
                )
        if self._configured_site_names is not None:
            raise RuntimeError("per-site config has already been applied to this recipe")
        if self._client_apps_prepared:
            raise RuntimeError(
                "per-site config must be applied immediately after recipe construction and before client "
                "configuration, files, filters, or components are added"
            )

        warn_on_potential_secrets(config, context="per_site_config")
        if self._SUPPORTED_PER_SITE_SECRET_REF_KEYS is not None:
            warn_on_unsupported_secret_refs_outside_keys(
                config,
                supported_value_keys=self._SUPPORTED_PER_SITE_SECRET_REF_KEYS,
                supported_value_depth=2,
                context="per_site_config",
            )
        else:
            warn_on_unsupported_secret_refs(config, context="per_site_config")
        # Copy each site's dictionary so deferred client preparation cannot be
        # changed by later mutation of the caller's config. Values such as data
        # loader objects intentionally retain their identity.
        config_snapshot = {site_name: dict(site_config) for site_name, site_config in config.items()}
        configured_site_names = tuple(config_snapshot)
        self._apply_per_site_config(config_snapshot)
        self._configured_site_names = configured_site_names

    def _apply_per_site_config(self, config: Dict[str, Dict]) -> None:
        """Validate and store recipe-specific per-site configuration.

        Client apps must not be added here. Recipes with configurable client
        topology should create them in ``_prepare_client_apps`` instead.
        """
        pass

    def _prepare_client_apps(self) -> None:
        """Create this recipe's client apps before client customization or use."""
        pass

    def _ensure_client_apps_prepared(self) -> None:
        """Prepare client apps once, after per-site configuration is known."""
        if self._client_apps_prepared:
            return

        self._prepare_client_apps()
        self._client_apps_prepared = True

    def _validate_before_use(self) -> None:
        """Validate recipe state immediately before export or execution."""
        pass

    def configured_sites(self) -> List[str]:
        """Return site keys configured through the helper or legacy constructor config.

        This reports configured site names only. It does not infer sites from job
        metadata, validate production enrollment, or indicate which clients are
        connected in the execution environment.
        """
        configured_site_names = getattr(self, "_configured_site_names", None)
        if configured_site_names is not None:
            return list(configured_site_names)

        legacy_per_site_config = getattr(self, "per_site_config", None)
        if isinstance(legacy_per_site_config, dict):
            return list(legacy_per_site_config.keys())

        return []

    def _snapshot_additional_params(self) -> Dict[str, Dict]:
        snapshot = {}
        deploy_map = getattr(self._job, "_deploy_map", {})
        for target, app in deploy_map.items():
            app_config = getattr(app, "app_config", None)
            if app_config is None:
                continue
            params = getattr(app_config, "additional_params", None)
            if isinstance(params, dict):
                snapshot[target] = dict(params)
        return snapshot

    def _restore_additional_params(self, snapshot: Dict[str, Dict]) -> None:
        deploy_map = getattr(self._job, "_deploy_map", {})
        for target, app in deploy_map.items():
            app_config = getattr(app, "app_config", None)
            if app_config is None:
                continue
            params = getattr(app_config, "additional_params", None)
            if isinstance(params, dict):
                original = snapshot.get(target, {})
                params.clear()
                params.update(original)

    def _replace_additional_params_for_targets(self, targets: List[str], new_params: dict) -> None:
        deploy_map = getattr(self._job, "_deploy_map", {})
        for target in targets:
            app = deploy_map.get(target)
            if app is None:
                continue
            app_config = getattr(app, "app_config", None)
            if app_config is None:
                continue
            params = getattr(app_config, "additional_params", None)
            if isinstance(params, dict):
                params.clear()
                params.update(new_params)

    @contextmanager
    def _temporary_exec_params(
        self, server_exec_params: Optional[dict] = None, client_exec_params: Optional[dict] = None
    ):
        """Temporarily override per-target additional_params during execute/export.

        Semantics:
        - None: leave the target's existing additional_params unchanged.
        - non-empty dict: temporarily apply/merge those params for the target.
        - empty dict ({}): temporarily clear the target's additional_params for this call.

        Any original additional_params are restored when the context exits.
        """
        if server_exec_params:
            warn_on_potential_secrets(server_exec_params, context="server_exec_params")
            warn_on_unsupported_secret_ref_keys(server_exec_params, context="server_exec_params")
        if client_exec_params:
            warn_on_potential_secrets(client_exec_params, context="client_exec_params")
            warn_on_unsupported_secret_ref_keys(client_exec_params, context="client_exec_params")

        params_snapshot = None
        if server_exec_params is not None or client_exec_params is not None:
            params_snapshot = self._snapshot_additional_params()

        try:
            if server_exec_params is not None:
                if server_exec_params:
                    self._job.to_server(server_exec_params)
                else:
                    # Preserve the long-standing "empty dict means temporarily clear params"
                    # behavior rather than treating {} as a no-op.
                    self._replace_additional_params_for_targets(["server"], {})

            if client_exec_params is not None:
                if client_exec_params:
                    self._add_to_client_apps(client_exec_params)
                else:
                    client_targets = [target for target in getattr(self._job, "_deploy_map", {}) if target != "server"]
                    self._replace_additional_params_for_targets(client_targets, {})
            yield
        finally:
            if params_snapshot is not None:
                self._restore_additional_params(params_snapshot)

    def _add_to_client_apps(self, obj, clients: Optional[List[str]] = None, **kwargs):
        """Add an object to client apps, preserving existing per-site structure.

        Args:
            obj: Object to add to clients.
            clients: Optional list of specific client names. If None, applies to all clients.
            **kwargs: Extra options forwarded to `job.to()`/`job.to_clients()`.

        Raises:
            TypeError: If clients is not a list.
            ValueError: If clients is empty or contains a non-client name; if specific
                clients are targeted while the recipe's client app applies to all clients
                (per-site placement cannot be expressed in the generated job in that
                topology); or if clients names a site with no existing client app while
                per-site client apps exist (that would deploy a bare, executor-less app
                to that site).
        """
        from nvflare.apis.job_def import ALL_SITES, SERVER_SITE_NAME
        from nvflare.job_config.defs import JobTargetType

        # Validate the selector before materializing topology so an invalid call
        # does not close the set_per_site_config() configuration window.
        if clients is not None:
            if not isinstance(clients, list):
                raise TypeError(f"clients must be a list of client names, got {type(clients).__name__}")
            if not clients:
                raise ValueError("clients must not be empty; omit it to apply to all clients")
            for client in clients:
                if not isinstance(client, str) or client in (ALL_SITES, SERVER_SITE_NAME):
                    raise ValueError(f"invalid client name {client!r}: client names must name specific client sites")

        self._ensure_client_apps_prepared()

        # FedJob has no public API to list per-site deploy targets, so we inspect
        # private deploy map to preserve existing per-site client topology.
        deploy_map = getattr(self._job, "_deploy_map", {})
        existing_client_sites = [
            target
            for target in deploy_map.keys()
            if target not in [ALL_SITES, SERVER_SITE_NAME]
            and JobTargetType.get_target_type(target) == JobTargetType.CLIENT
        ]
        if clients is None:
            if existing_client_sites:
                for site in existing_client_sites:
                    self._job.to(obj, site, **kwargs)
            else:
                self._job.to_clients(obj, **kwargs)
        else:
            if ALL_SITES in deploy_map:
                # The generated job has one client app deployed to all clients. Exporting
                # both an all-clients app and per-site apps is not expressible (the
                # all-clients app wins and per-site apps are dropped), so fail loudly
                # instead of silently losing the placement.
                raise ValueError(
                    "cannot target specific clients: this recipe's client app applies to all clients. "
                    "Call set_per_site_config immediately after constructing a recipe that supports it, "
                    "or omit clients to apply to all clients."
                )
            if existing_client_sites:
                # Targeting a site with no app would create a bare, executor-less app for
                # that site in the exported job — the same class of quiet misconfiguration
                # as the ALL_SITES case above, so fail loudly instead.
                unknown = [c for c in clients if c not in existing_client_sites]
                if unknown:
                    raise ValueError(
                        f"unknown client site(s) {unknown}: this recipe has per-site client apps "
                        f"only for {sorted(existing_client_sites)}"
                    )
            for client in clients:
                self._job.to(obj, client, **kwargs)

    def add_client_input_filter(
        self, filter: Filter, tasks: Optional[List[str]] = None, clients: Optional[List[str]] = None
    ):
        """Add a filter to clients for incoming tasks from the server.

        Args:
            filter: the filter to be added
            tasks: tasks that the filter applies to
            clients: client names to add, if None, all clients will be added.

        Returns: None

        """
        self._add_to_client_apps(filter, clients=clients, filter_type=FilterType.TASK_DATA, tasks=tasks)

    def add_client_output_filter(
        self, filter: Filter, tasks: Optional[List[str]] = None, clients: Optional[List[str]] = None
    ):
        """Add a filter to clients for outgoing result to server.

        Args:
            filter: the filter to be added
            tasks: tasks that the filter applies to
            clients: client names to add, if None, all clients will be added.

        Returns: None

        """
        self._add_to_client_apps(filter, clients=clients, filter_type=FilterType.TASK_RESULT, tasks=tasks)

    def add_client_config(self, config: Dict, clients: Optional[List[str]] = None):
        """Add top-level configuration parameters to config_fed_client.json.

        The config values are written in clear text into the generated
        ``config_fed_client.json`` and must never contain actual secret values;
        a ``PotentialSecretWarning`` is emitted for values that look like secrets.
        See the Recipe class docstring for the recommended alternatives.

        Args:
            config: Dictionary of configuration parameters to add.
            clients: Optional list of specific client names. If None, applies to all clients.

        Raises:
            TypeError: If config is not a dictionary.
        """
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config).__name__}")

        warn_on_potential_secrets(config, context="add_client_config config")
        warn_on_unsupported_secret_ref_keys(config, context="add_client_config config")
        self._add_to_client_apps(config, clients=clients)

    def add_client_file(self, file_path: str, clients: Optional[List[str]] = None):
        """Add a file or directory to client apps.

        The file will be added to the client's custom directory and bundled with the job.
        Can be a script, configuration file, or any resource needed by clients.

        Args:
            file_path: Path to the file or directory to add to clients.
            clients: Optional list of specific client names. If None, applies to all clients.

        Raises:
            TypeError: If file_path is not a string.

        Example:
            # Add a wrapper script to all clients
            recipe.add_client_file("client_wrapper.sh")

            # Add a script to specific clients
            recipe.add_client_file("custom_script.py", clients=["site1", "site2"])
        """
        if not isinstance(file_path, str):
            raise TypeError(f"file_path must be a str, got {type(file_path).__name__}")

        self._add_to_client_apps(file_path, clients=clients)

    def add_server_output_filter(self, filter: Filter, tasks: Optional[List[str]] = None):
        """Add a filter to the server for outgoing tasks to clients.

        Args:
            filter: the filter to be added
            tasks: tasks that the filter applies to

        Returns: None

        """
        self._job.to_server(filter, filter_type=FilterType.TASK_DATA, tasks=tasks)

    def add_server_input_filter(self, filter: Filter, tasks: Optional[List[str]] = None):
        """Add a filter to server for incoming task result from clients. .

        Args:
            filter: the filter to be added
            tasks: tasks that the filter applies to

        Returns: None

        """
        self._job.to_server(filter, filter_type=FilterType.TASK_RESULT, tasks=tasks)

    def add_server_config(self, config: Dict):
        """Add top-level configuration parameters to config_fed_server.json.

        The config values are written in clear text into the generated
        ``config_fed_server.json`` and must never contain actual secret values;
        a ``PotentialSecretWarning`` is emitted for values that look like secrets.
        See the Recipe class docstring for the recommended alternatives.

        Args:
            config: Dictionary of configuration parameters to add.

        Raises:
            TypeError: If config is not a dictionary.
        """
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config).__name__}")

        warn_on_potential_secrets(config, context="add_server_config config")
        warn_on_unsupported_secret_ref_keys(config, context="add_server_config config")
        self._job.to_server(config)

    def add_server_file(self, file_path: str):
        """Add a file or directory to server app.

        The file will be added to the server's custom directory and bundled with the job.
        Can be a script, configuration file, or any resource needed by the server.

        Args:
            file_path: Path to the file or directory to add to server.

        Raises:
            TypeError: If file_path is not a string.

        Example:
            # Add a wrapper script to server
            recipe.add_server_file("server_wrapper.sh")
        """
        if not isinstance(file_path, str):
            raise TypeError(f"file_path must be a str, got {type(file_path).__name__}")

        self._job.to_server(file_path)

    @staticmethod
    def _get_full_class_name(obj):
        """
        Returns the fully qualified name of an object.
        """
        cls = type(obj)
        module = cls.__module__
        qualname = cls.__qualname__
        if module == "builtins":  # For built-in types like int, str, etc.
            return qualname
        return f"{module}.{qualname}"

    def enable_log_streaming(self, *file_names: str) -> None:
        """Enable live log streaming from clients to the server while the job runs.

        Adds one ``JobLogStreamer`` per file name to clients and a single
        ``JobLogReceiver`` to the server. Streaming is still gated per-site by
        ``allow_log_streaming`` in ``resources.json``.

        Args:
            *file_names: log file base names to stream. If omitted, defaults to
                ``"log.json"``.

        Example::

            recipe.enable_log_streaming()                     # streams log.json
            recipe.enable_log_streaming("log.txt")            # streams a single file
            recipe.enable_log_streaming("log.json", "log.txt")  # streams both
        """
        from nvflare.app_common.logging.job_log_receiver import JobLogReceiver
        from nvflare.app_common.logging.job_log_streamer import JobLogStreamer

        if not file_names:
            file_names = ("log.json",)

        for name in file_names:
            self._add_to_client_apps(JobLogStreamer(log_file_name=name))
        self._job.to_server(JobLogReceiver())

    def enable_tensor_streaming(
        self,
        format: str = "pytorch",
        tasks: Optional[List[str]] = None,
        tensor_send_timeout: float = 30.0,
        wait_send_task_data_all_clients_timeout: float = 300.0,
    ) -> None:
        """Enable tensor streaming between the server and all client apps.

        The same exchange format, task names, and per-transfer timeout are used
        on both sides. The format must match the recipe's
        ``server_expected_format`` when the recipe declares one.

        Args:
            format: Tensor exchange format. Defaults to ``"pytorch"``
                (``ExchangeFormat.PYTORCH``).
            tasks: Task names whose tensors should be streamed. ``None`` uses
                the streamers' default ``["train"]``.
            tensor_send_timeout: Timeout in seconds for each tensor transfer.
            wait_send_task_data_all_clients_timeout: Maximum time in seconds
                for the server to wait for all clients to receive task tensors.

        Raises:
            TypeError: If ``tasks`` is not a list of strings.
            ValueError: If ``tasks`` is empty, if ``format`` does not match a
                declared ``server_expected_format``.
            RuntimeError: If tensor streaming was already enabled.
        """
        from nvflare.app_opt.tensor_stream.client import TensorClientStreamer
        from nvflare.app_opt.tensor_stream.server import TensorServerStreamer

        if getattr(self, "_tensor_streaming_added", False):
            raise RuntimeError("tensor streaming has already been enabled for this recipe")

        if tasks is not None:
            if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
                raise TypeError(f"tasks must be a list of str, got {tasks!r}")
            if not tasks:
                raise ValueError("tasks must not be empty; use None for the default train task")

        server_expected_format = getattr(self, "server_expected_format", None)
        if server_expected_format is not None and format != server_expected_format:
            raise ValueError(
                f"tensor streaming format {format!r} must match server_expected_format {server_expected_format!r}"
            )

        server_tasks = list(tasks) if tasks is not None else None
        client_tasks = list(tasks) if tasks is not None else None
        self._job.to_server(
            TensorServerStreamer(
                format=format,
                tasks=server_tasks,
                tensor_send_timeout=tensor_send_timeout,
                wait_send_task_data_all_clients_timeout=wait_send_task_data_all_clients_timeout,
            ),
            id="tensor_server_streamer",
        )
        self._add_to_client_apps(
            TensorClientStreamer(
                format=format,
                tasks=client_tasks,
                tensor_send_timeout=tensor_send_timeout,
            ),
            id="tensor_client_streamer",
        )
        self._tensor_streaming_added = True

    def add_decomposers(self, decomposers: List[Union[str, Decomposer]]):
        """Add decomposers to the job

        Args:
            decomposers: spec of decomposers. Can be class names or Decomposer objects

        Returns: None

        """
        if not decomposers:
            return

        class_names = []
        for d in decomposers:
            if isinstance(d, str):
                # class name
                class_names.append(d)
            elif isinstance(d, Decomposer):
                class_names.append(self._get_full_class_name(d))
            else:
                raise TypeError(f"decomposer must be str or Decomposer, got {type(d).__name__}")

        self._job.to_server(DecomposerRegister(class_names), id="decomposer_reg")
        self._add_to_client_apps(DecomposerRegister(class_names), id="decomposer_reg")

    def _warn_potential_secrets_in_exported_job(self, job_dir: str) -> None:
        """Scan the exported job's generated config files for secret-looking values.

        This is a last-line best-effort check on the artifacts actually produced;
        it does not redact values or make a job folder safe. Only generated JSON
        config files are scanned -- user code bundled under ``custom/`` is not.
        """
        warn_on_potential_secrets_in_job_dir(
            job_dir=job_dir,
            job_name=getattr(self._job, "name", None),
            context="exported job file",
        )

    def export(
        self,
        job_dir: str,
        server_exec_params: Optional[dict] = None,
        client_exec_params: Optional[dict] = None,
        env: Optional[ExecEnv] = None,
    ):
        """Export the recipe to a job definition.

        Recipe parameters can appear in the exported job folder in clear text. Generated
        config files are scanned and a ``PotentialSecretWarning`` is emitted for values that
        look like actual secrets. This best-effort scan does not redact values or prove that
        an export is safe; callers must follow the Recipe no-secret contract. See the Recipe
        class docstring for how to pass references instead.

        Args:
            job_dir: directory where the job will be exported to.
            server_exec_params: execution params for the server
            client_exec_params: execution params for clients
            env: the environment that the exported job will be running in

        Returns: None

        """
        self._validate_before_use()
        self._ensure_client_apps_prepared()
        self._warn_potential_secrets_in_params()
        with self._temporary_exec_params(server_exec_params=server_exec_params, client_exec_params=client_exec_params):
            if env is not None:
                self.process_env(env)
            self._job.export_job(job_dir)
        self._warn_potential_secrets_in_exported_job(job_dir)

    def run(
        self, env: ExecEnv, server_exec_params: Optional[dict] = None, client_exec_params: Optional[dict] = None
    ) -> "Run":
        """Run the recipe in a specified execution environment.

        Args:
            env: the execution environment
            server_exec_params: execution params for the server
            client_exec_params: execution params for clients

        Returns: Run to get job ID and execution results

        """
        self._validate_before_use()
        self._ensure_client_apps_prepared()
        self._warn_potential_secrets_in_params()
        with self._temporary_exec_params(server_exec_params=server_exec_params, client_exec_params=client_exec_params):
            self.process_env(env)
            job_id = env.deploy(self._job)
            from nvflare.recipe.run import Run

            return Run(env, job_id)

    def execute(
        self,
        env: ExecEnv,
        server_exec_params: Optional[dict] = None,
        client_exec_params: Optional[dict] = None,
    ) -> Optional["Run"]:
        """Execute or export the recipe based on command-line flags.

        Transparently checks sys.argv for ``--export`` / ``--export-dir`` without
        interfering with the caller's own argument parser.

        * ``python job.py``                         → run the job
        * ``python job.py --export``                → export to ``./fl_job``
        * ``python job.py --export --export-dir X`` → export to ``X``

        Args:
            env: the execution environment
            server_exec_params: execution params for the server
            client_exec_params: execution params for clients

        Returns:
            Run when executing; raises SystemExit(0) when exporting so callers
            need not guard against a None return value.
        """
        recipe_export, recipe_export_dir = _peek_recipe_args()
        if recipe_export:
            self.export(
                job_dir=recipe_export_dir,
                server_exec_params=server_exec_params,
                client_exec_params=client_exec_params,
                env=env,
            )
            print(f"Job exported to: {recipe_export_dir}")
            raise SystemExit(0)

        return self.run(env, server_exec_params=server_exec_params, client_exec_params=client_exec_params)

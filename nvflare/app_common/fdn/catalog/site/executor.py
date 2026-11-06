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

"""The client-side executor of the catalog.read task."""

import logging
import os
import time
from dataclasses import replace
from pathlib import Path

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import get_job_meta_study
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.fdn.catalog.common.payload import (
    CATALOG_READ_KEY,
    TASK_NAME,
    PayloadError,
    ReasonCode,
    ResultStatus,
    build_result,
    parse_task_payload,
)
from nvflare.app_common.fdn.catalog.common.schema import ValidationError, validate
from nvflare.app_common.fdn.catalog.site.connector import CatalogConnector, ConnectorError
from nvflare.app_common.fdn.catalog.site.policy import PolicyError, apply_policy, load_policy
from nvflare.app_common.fdn.catalog.site.site_config import (
    DATASET_NAME,
    SITE_CONFIG_FILE,
    DatasetBinding,
    SiteConfigError,
    load_site_config,
)
from nvflare.app_opt.job_launcher.study_runtime import (
    load_study_runtime_file,
    resolve_study_runtime,
    study_runtime_file_path,
)
from nvflare.fuel.utils.class_loader import load_class

DEFAULT_MOUNT_ROOT = "/data"
DEFAULT_POLICY_ROOT = "/fdn/policy"


_logger = logging.getLogger(__name__)


class MountResolver:
    """The Study's current dataset bindings.

    Container launchers mount each dataset at <mount_root>/<study>/<dataset>. Process mode mounts nothing, so the
    bindings come from the `datasets` of the Study's entry in the site's study_runtime.yaml, where `source` is the
    real path.
    """

    def __init__(self, mount_root: str):
        self._root = Path(mount_root)

    def resolve(self, study: str, runtime_file: str | None = None) -> dict[str, Path]:
        study_dir = self._root / study
        if study_dir.is_dir():
            return {
                p.name: p
                for p in sorted(study_dir.iterdir())
                if (p.is_dir() or p.is_file()) and DATASET_NAME.match(p.name)
            }
        if runtime_file is None or not os.path.exists(runtime_file):
            return {}
        try:
            runtime = resolve_study_runtime(load_study_runtime_file(runtime_file, "process"), study, runtime_file)
        except (OSError, ValueError) as e:
            _logger.warning("the study runtime file could not be used, so no dataset is bound: %s", type(e).__name__)
            return {}
        paths = {d.dataset: Path(d.source) for d in runtime.datasets}
        return {
            name: path
            for name, path in sorted(paths.items())
            if (path.is_dir() or path.is_file()) and DATASET_NAME.match(name)
        }


class CatalogExecutor(Executor):
    def __init__(self):
        """No arguments, so a job cannot configure it. The connector, its limits and the policy come from the
        site's own files under the policy root, and the data mounts from the fixed mount root.
        """
        super().__init__()
        self.policy_root: Path | None = None  # tests set it; otherwise it is resolved per read
        self.resolver = MountResolver(DEFAULT_MOUNT_ROOT)

    @staticmethod
    def _workspace_root(fl_ctx: FLContext) -> str | None:
        engine = fl_ctx.get_engine()
        workspace = engine.get_workspace() if engine is not None else None
        return workspace.get_root_dir() if workspace is not None else None

    def _policy_root(self, workspace_root: str | None) -> Path:
        """The read-only mount that container launchers expose, else the site's local/fdn (process mode)."""
        if self.policy_root is not None:
            return self.policy_root
        if Path(DEFAULT_POLICY_ROOT).is_dir() or workspace_root is None:
            return Path(DEFAULT_POLICY_ROOT)
        return Path(workspace_root) / "local" / "fdn"

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name != TASK_NAME:
            return make_reply(ReturnCode.TASK_UNKNOWN)
        return self._reply(self._read(shareable, fl_ctx, abort_signal))

    @staticmethod
    def _without_unshared_lineage(catalog, policy):
        """Lineage must not name a dataset that policy does not share."""
        shared = tuple(x for x in catalog.lineage if x.dataset in policy.datasets and policy.datasets[x.dataset].share)
        return catalog if len(shared) == len(catalog.lineage) else replace(catalog, lineage=shared)

    @staticmethod
    def _reply(result: dict) -> Shareable:
        reply = make_reply(ReturnCode.OK)
        reply[CATALOG_READ_KEY] = result
        return reply

    @staticmethod
    def _fail(selected: list[str], previous: set[str], reason: ReasonCode) -> dict:
        """Nothing can be shared, and the permitted view of an indexed dataset cannot be rebuilt, so every selected
        name the server already holds is withheld until a later read succeeds."""
        return build_result(ResultStatus.FAILED, withheld=[n for n in selected if n in previous], reason_code=reason)

    def _open_connectors(self, names, site, mounts: dict[str, Path], fl_ctx: FLContext) -> dict:
        """One instance per connector this Study's datasets use; None for one that cannot be built, so it affects
        only its own datasets."""
        opened = {}
        for name in sorted(names):
            spec = site.connectors[name]
            try:
                connector_class = load_class(spec.path)
                if not (isinstance(connector_class, type) and issubclass(connector_class, CatalogConnector)):
                    raise TypeError("not a CatalogConnector")
                opened[name] = connector_class(mounts, **spec.args)
            except Exception as e:
                opened[name] = None
                self.log_warning(fl_ctx, f"catalog connector '{name}' could not be loaded: {type(e).__name__}")
        return opened

    def _read(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> dict:
        try:
            task = parse_task_payload(shareable.get(CATALOG_READ_KEY))
        except PayloadError:
            return build_result(ResultStatus.FAILED, reason_code=ReasonCode.INTERNAL)

        study = get_job_meta_study(fl_ctx.get_prop(FLContextKey.JOB_META))
        workspace_root = self._workspace_root(fl_ctx)
        runtime_file = study_runtime_file_path(workspace_root) if workspace_root else None
        previous = set(task.previous)
        asked = list(task.datasets) if task.datasets is not None else sorted(previous)
        try:
            mounts = self.resolver.resolve(study, runtime_file)
        except OSError as e:
            self.log_warning(fl_ctx, f"catalog datasets could not be listed: {type(e).__name__}")
            return self._fail(asked, previous, ReasonCode.INTERNAL)

        policy_root = self._policy_root(workspace_root)
        try:
            site = load_site_config(policy_root / SITE_CONFIG_FILE)
            policy = load_policy(policy_root / f"{study}.json")
        except (PolicyError, SiteConfigError):
            return self._fail(asked, previous, ReasonCode.POLICY_UNAVAILABLE)

        limits = site.limits
        deadline = time.monotonic() + limits.max_duration_s
        # The Study's datasets: those the file binds to a connector, and the mounted ones, which the default
        # connector reads unless the file binds them elsewhere.
        bindings = site.bindings(study)
        routes = {name: DatasetBinding(site.default_connector) for name in mounts}
        routes.update(bindings)
        selected = asked if task.datasets is not None else sorted(set(routes) | previous)
        needed = {
            routes[name].connector
            for name in selected
            if name in routes and name in policy.datasets and policy.datasets[name].share
        }
        opened = self._open_connectors(needed, site, mounts, fl_ctx)
        withheld, unavailable, catalogs = [], [], []
        reason = None
        used_records = 0
        try:
            if opened and not any(opened.values()):
                return self._fail(asked, previous, ReasonCode.INTERNAL)
            if None in opened.values():
                reason = ReasonCode.INTERNAL  # some connector is down; the other datasets are still read
            for name in selected:
                rule = policy.datasets.get(name)
                if rule is None or not rule.share:
                    # Only a previously indexed name is reported: naming a never-shared mount would disclose it.
                    if name in previous:
                        withheld.append(name)
                    continue
                if abort_signal.triggered or time.monotonic() >= deadline or used_records >= limits.max_records:
                    reason = ReasonCode.CANCELLED if abort_signal.triggered else ReasonCode.BUDGET_EXCEEDED
                    if name in previous:
                        withheld.append(name)
                    continue
                route = routes.get(name)
                connector = opened[route.connector] if route is not None else None
                if connector is None:
                    unavailable.append(name)  # no connector reads it now, or its connector is down
                    continue
                try:
                    per_dataset = replace(
                        limits,
                        max_duration_s=min(limits.max_duration_s, max(1, int(deadline - time.monotonic()))),
                        max_records=limits.max_records - used_records,
                    )
                    catalog = apply_policy(
                        rule, connector.read_catalog(name, per_dataset, route.options), site.installed_operations
                    )
                    if catalog.dataset != name:
                        raise ValidationError("a connector may only return the dataset it was asked for")
                    catalog = self._without_unshared_lineage(catalog, policy)
                    if catalog.record_count() > per_dataset.max_records:
                        raise ConnectorError("record budget exceeded", budget=True)
                    validate(catalog, per_dataset)
                    used_records += catalog.record_count()
                    catalogs.append(catalog)
                except ConnectorError as e:
                    if not e.budget:
                        unavailable.append(name)
                    else:
                        reason = ReasonCode.BUDGET_EXCEEDED
                        if name in previous:
                            withheld.append(name)
                except ValidationError:
                    unavailable.append(name)
                except Exception as e:
                    self.log_warning(fl_ctx, f"catalog read failed for a dataset: {type(e).__name__}")
                    unavailable.append(name)
        finally:
            for connector in filter(None, opened.values()):
                try:
                    connector.close()
                except Exception as e:
                    self.log_warning(fl_ctx, f"catalog connector close failed: {type(e).__name__}")

        # A completed policy-only read is authoritative even when no dataset has ever been shared.
        # Its coverage must prevent an older, more permissive result from first publishing a name.
        status = ResultStatus.OK if catalogs or withheld or unavailable or reason is None else ResultStatus.FAILED
        return build_result(status, catalogs=catalogs, withheld=withheld, unavailable=unavailable, reason_code=reason)

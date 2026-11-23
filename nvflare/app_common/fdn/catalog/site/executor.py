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

"""The site-side executor of the catalog_read task."""

import time
from dataclasses import replace

from nvflare.apis.dataset_config_spec import DatasetConfigError, StudyDatasetConfig, get_study_dataset_config
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.fdn.catalog.common.payload import (
    CATALOG_JOB_META,
    CATALOG_READ_KEY,
    TASK_NAME,
    PayloadError,
    ReasonCode,
    ResultStatus,
    build_result,
    parse_task_payload,
)
from nvflare.app_common.fdn.catalog.common.schema import ValidationError, validate
from nvflare.app_common.fdn.catalog.site.connector import BUILTIN_CONNECTORS, CatalogConnector, ConnectorError
from nvflare.app_common.fdn.catalog.site.settings import (
    CATALOG_COMPONENT,
    CatalogSettings,
    ConnectorSpec,
    SettingsError,
    parse_settings,
)
from nvflare.fuel.utils.class_loader import load_class


class CatalogExecutor(Executor):
    """Reads the job's Study datasets through the connectors the site's dataset configuration selects."""

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name != TASK_NAME:
            return make_reply(ReturnCode.TASK_UNKNOWN)
        meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        if not isinstance(meta, dict) or meta.get(CATALOG_JOB_META) is not True:
            return make_reply(ReturnCode.TASK_UNSUPPORTED)
        result = self._read(shareable, fl_ctx, abort_signal)
        if result is None:
            return make_reply(ReturnCode.TASK_ABORTED)
        reply = make_reply(ReturnCode.OK)
        reply[CATALOG_READ_KEY] = result
        return reply

    def load_config(self, fl_ctx: FLContext) -> StudyDatasetConfig:
        """The dataset configuration snapshot of the job's Study."""
        return get_study_dataset_config(fl_ctx)

    def _connector_for(self, name: str, settings: CatalogSettings, fl_ctx: FLContext, cache: dict):
        """The connector instance for a connector name, or None when it cannot be built. One instance per name."""
        if name in cache:
            return cache[name]
        spec = settings.connectors.get(name)
        if spec is None and name in BUILTIN_CONNECTORS:
            spec = ConnectorSpec(BUILTIN_CONNECTORS[name])
        connector, problem = (None, "it is neither built in nor declared") if spec is None else self._build(spec)
        if connector is None:
            # Only the site's own log names the cause; the result carries no detail.
            self.log_error(
                fl_ctx,
                f"catalog connector '{name}' cannot be used: {problem}. Fix usage.catalog in the site dataset "
                "configuration; its datasets stay unavailable until then.",
            )
        cache[name] = connector
        return connector

    @staticmethod
    def _build(spec: ConnectorSpec):
        """The connector instance, or None with the reason it cannot be built. Exception messages are not
        included, as they may carry paths or settings."""
        try:
            connector_class = load_class(spec.path)
        except Exception:
            return None, "the class cannot be imported or found"
        if not (isinstance(connector_class, type) and issubclass(connector_class, CatalogConnector)):
            return None, "the class is not a CatalogConnector subclass"
        try:
            return connector_class(**spec.args), None
        except Exception as e:
            return None, f"its constructor failed ({type(e).__name__}); check the connector's args"

    def _read(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> dict | None:
        try:
            task = parse_task_payload(shareable.get(CATALOG_READ_KEY))
        except PayloadError:
            return build_result(ResultStatus.FAILED, reason_code=ReasonCode.INTERNAL)
        try:
            config = self.load_config(fl_ctx)
            settings = parse_settings(config.site_usage(CATALOG_COMPONENT))
        except (DatasetConfigError, SettingsError) as e:
            self.log_error(fl_ctx, f"catalog read refused: {e}")
            return build_result(ResultStatus.FAILED, reason_code=ReasonCode.CONFIG_UNAVAILABLE)

        listed = config.list_datasets()
        # A catalog can name other datasets through lineage, even on a selective read.
        # Refuse an unmigrated Study before returning any identifiers.
        if any(config.get_dataset(name).usage_rules(CATALOG_COMPONENT) is not None for name in listed):
            self.log_error(
                fl_ctx,
                "per-dataset usage.catalog is no longer supported; configure source metadata permissions "
                "and remove the obsolete catalog rules from the site dataset configuration",
            )
            return build_result(ResultStatus.FAILED, reason_code=ReasonCode.CONFIG_UNAVAILABLE)
        selected = [n for n in task.datasets if n in listed] if task.datasets is not None else listed
        limits = settings.limits
        # The read ends by the site's own limit or the time the task leaves, whichever is first, so completed catalogs
        # are returned before the controller gives up on this site.
        duration = limits.max_duration_s
        if task.time_budget_s is not None:
            duration = min(duration, task.time_budget_s)
        deadline = time.monotonic() + duration
        catalogs, unavailable, skipped, opened = [], [], [], {}
        used_records = 0
        try:
            for name in selected:
                if abort_signal.triggered:
                    return None
                if time.monotonic() >= deadline or used_records >= limits.max_records:
                    skipped.append(name)
                    continue
                dataset = config.get_dataset(name)
                connector = self._connector_for(dataset.type, settings, fl_ctx, opened)
                if connector is None:
                    unavailable.append(name)
                    continue
                remaining = replace(
                    limits,
                    max_duration_s=max(0.0, deadline - time.monotonic()),
                    max_records=limits.max_records - used_records,
                )
                try:
                    catalog = connector.read_catalog(dataset, lambda key, n=name: config.get_secret(n, key), remaining)
                    if time.monotonic() >= deadline:
                        raise ConnectorError("read budget exceeded", budget=True)
                    if catalog.dataset != name:
                        raise ValidationError("a connector may only return the dataset it was asked for")
                    # Advertising an operation is limited to installed capabilities; it grants no execution rights.
                    catalog = replace(
                        catalog, operations=tuple(sorted(set(catalog.operations) & settings.installed_operations))
                    )
                    if catalog.record_count() > remaining.max_records:
                        raise ConnectorError("record budget exceeded", budget=True)
                    validate(catalog, remaining)
                    used_records += catalog.record_count()
                    catalogs.append(catalog)
                except ConnectorError as e:
                    (skipped if e.budget else unavailable).append(name)
                except (ValidationError, DatasetConfigError):
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
        return build_result(ResultStatus.OK, catalogs=catalogs, unavailable=unavailable, skipped=skipped)

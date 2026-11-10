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
from collections.abc import Mapping
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
from nvflare.app_common.fdn.catalog.site.rules import (
    CATALOG_COMPONENT,
    CatalogSettings,
    ConnectorSpec,
    RulesError,
    apply_policy,
    parse_rule,
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
        except (DatasetConfigError, RulesError) as e:
            self.log_error(fl_ctx, f"catalog read refused: {e}")
            return build_result(ResultStatus.FAILED, reason_code=ReasonCode.CONFIG_UNAVAILABLE)

        listed = config.list_datasets()
        selected = [n for n in task.datasets if n in listed] if task.datasets is not None else listed
        rules, unavailable = {}, []
        for name in listed:  # every rule is checked, so lineage can be filtered against what the site shares
            raw = config.get_dataset(name).usage_rules(CATALOG_COMPONENT)
            if raw is None:
                continue
            # Only a dataset whose rules explicitly share it may ever be named outside the site, in the result or in
            # a log line (job logs can be streamed to the server).
            explicitly_shared = isinstance(raw, Mapping) and raw.get("share") is True
            label = f"datasets.{name}.usage.catalog" if explicitly_shared else "a dataset's usage.catalog"
            try:
                rules[name] = parse_rule(raw, label)
            except RulesError as e:
                # A rule error never names table or column names; the dataset is named only if explicitly shared.
                self.log_error(fl_ctx, f"catalog rules are invalid and the dataset is not read: {e}")
                if explicitly_shared and name in selected:
                    unavailable.append(name)
        shared = {name for name, rule in rules.items() if rule.share}

        limits = settings.limits
        # The read ends by the site's own limit or the time the task leaves, whichever is first, so completed catalogs
        # are returned before the controller gives up on this site.
        duration = limits.max_duration_s
        if task.time_budget_s is not None:
            duration = min(duration, task.time_budget_s)
        deadline = time.monotonic() + duration
        catalogs, skipped, opened = [], [], {}
        used_records = 0
        try:
            for name in selected:
                if name not in shared:
                    continue  # not shared: the name never leaves the site
                if abort_signal.triggered:
                    return None
                if time.monotonic() >= deadline or used_records >= limits.max_records:
                    skipped.append(name)
                    continue
                dataset = config.get_dataset(name)
                connector = self._connector_for(rules[name].connector or dataset.type, settings, fl_ctx, opened)
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
                    catalog = apply_policy(rules[name], catalog, settings.installed_operations)
                    # lineage never names a dataset the site does not share
                    catalog = replace(catalog, lineage=tuple(x for x in catalog.lineage if x.dataset in shared))
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

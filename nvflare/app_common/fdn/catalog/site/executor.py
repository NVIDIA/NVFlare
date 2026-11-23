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

from dataclasses import replace

from nvflare.apis.dataset_config_spec import DatasetConfigError, StudyDatasetConfig, get_study_dataset_config
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.fdn.catalog.common.odcs import check_contract
from nvflare.app_common.fdn.catalog.common.payload import (
    CATALOG_JOB_META,
    CATALOG_READ_KEY,
    MAX_SITE_REPLY_BYTES,
    TASK_NAME,
    PayloadError,
    ReasonCode,
    ResultStatus,
    build_result,
    json_size,
    parse_task_payload,
)
from nvflare.app_common.fdn.catalog.site.settings import CATALOG_COMPONENT, SettingsError, parse_settings
from nvflare.app_common.fdn.catalog.site.source_error import CatalogSourceError
from nvflare.app_opt.fdn.catalog._metadata import ReadBudget
from nvflare.app_opt.fdn.catalog.catalog_discovery_worker import run_catalog_discovery


class CatalogExecutor(Executor):
    """Read ODCS contracts from the job's site-owned Study dataset configuration."""

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
        # Fail closed on obsolete publication rules before disclosing any dataset name.
        if any(config.get_dataset(name).usage_rules(CATALOG_COMPONENT) for name in listed):
            self.log_error(
                fl_ctx,
                "obsolete usage.catalog settings; configure source metadata permissions "
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
        budget = ReadBudget(replace(limits, max_duration_s=duration))
        catalogs, unavailable, skipped = [], [], []
        # Reserve a status entry for every selection before spending space on successful reads.
        bytes_left = MAX_SITE_REPLY_BYTES - json_size(build_result(ResultStatus.OK, skipped=selected))
        if bytes_left < 0:
            return build_result(ResultStatus.FAILED, reason_code=ReasonCode.INTERNAL)
        for name in selected:
            if abort_signal.triggered:
                return None
            if not budget.calls_left or not budget.records_left:
                skipped.append(name)
                continue
            dataset = config.get_dataset(name)
            try:
                budget.remaining()
                contract = run_catalog_discovery(
                    dataset, lambda key, n=name: config.get_secret(n, key), budget, site=fl_ctx.get_identity_name()
                )
                records = check_contract(contract)
                budget.records(records)
                entry = {"dataset": name, "contract": contract}
                size = json_size(entry) + 2  # separator between entries
                if size > bytes_left:
                    raise CatalogSourceError("site reply byte budget exceeded", budget=True)
                bytes_left -= size
                budget.records_left -= records
                catalogs.append(entry)
            except CatalogSourceError as e:
                (skipped if e.budget else unavailable).append(name)
            except Exception as e:
                self.log_warning(fl_ctx, f"catalog read failed for a dataset: {type(e).__name__}")
                unavailable.append(name)
        return build_result(ResultStatus.OK, catalogs=catalogs, unavailable=unavailable, skipped=skipped)

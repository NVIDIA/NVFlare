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

import json
import math
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.utils.file_utils import resolve_path_under_root
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.widgets.widget import Widget

METRICS_AGGREGATION_INFO = AppConstants.METRICS_AGGREGATION_INFO


class MetricsArtifactWriter(Widget):
    """Writes safe, machine-readable round and summary metric artifacts.

    The writer consumes metrics already produced by workflows/controllers. It records
    dynamic metric names as values instead of object keys so downstream consumers do
    not need to treat client-provided names as JSON object structure.
    """

    def __init__(
        self,
        results_dir: str = "metrics",
        summary_file_name: str = "metrics_summary.json",
        round_file_name: str = "round_metrics.jsonl",
        limits: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.results_dir = results_dir
        self.summary_file_name = summary_file_name
        self.round_file_name = round_file_name

        limits = limits or {}
        self.max_metric_name_length = limits.get("max_metric_name_length", 256)
        self.max_string_value_length = limits.get("max_string_value_length", 1024)
        self.max_metrics_per_site_per_round = limits.get("max_metrics_per_site_per_round", 512)
        self.max_sites_per_round = limits.get("max_sites_per_round", 10000)
        self.max_site_metric_records_per_round = limits.get("max_site_metric_records_per_round", 10000)
        self.max_skipped_metrics_per_round = limits.get("max_skipped_metrics_per_round", 1024)
        self.max_round_record_bytes = limits.get("max_round_record_bytes", 1048576)
        self.max_summary_bytes = limits.get("max_summary_bytes", 1048576)
        self.max_int_bit_length = limits.get("max_int_bit_length", 1023)

        self._reset()

    def _reset(self):
        self._has_metrics = False
        self._final_round = None
        self._final_aggregated_metrics = []
        self._best_selection = None
        self._key_metric = None
        self._aggregation = None
        self._metric_source = None
        self._metric_split = None
        self._round_file_path = None
        self._summary_file_path = None
        self._round_sites = {}
        self._round_skipped = {}
        self._round_site_metric_counts = {}

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._reset()
        elif event_type == AppEventType.AFTER_CONTRIBUTION_ACCEPT:
            self._handle_after_contribution_accept(fl_ctx)
        elif event_type == AppEventType.AFTER_AGGREGATION:
            self._handle_after_aggregation(fl_ctx)
        elif event_type == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE:
            self._handle_global_best_model_available(fl_ctx)
        elif event_type == EventType.END_RUN:
            self._write_summary_if_needed(fl_ctx)

    def _handle_after_aggregation(self, fl_ctx: FLContext):
        aggr_result = fl_ctx.get_prop(AppConstants.AGGREGATION_RESULT, None)
        aggr_result = self._to_fl_model(aggr_result)
        if aggr_result is None:
            return

        meta = aggr_result.meta or {}
        info = meta.get(METRICS_AGGREGATION_INFO, {})
        if not isinstance(info, dict):
            info = {}

        skipped = []
        aggregated_metrics = self._normalize_metrics(
            aggr_result.metrics, site=None, skipped=skipped, for_aggregation=True
        )
        current_round = self._get_current_round(aggr_result, fl_ctx)
        fallback_sites = self._round_sites.pop(current_round, [])
        fallback_skipped = self._round_skipped.pop(current_round, [])
        self._round_site_metric_counts.pop(current_round, None)
        sites = self._normalize_sites(info.get("sites"), skipped)
        site_weights = self._normalize_site_weights(info.get("site_weights"), skipped)
        use_contribution_sites = info.get("use_contribution_sites", True) is not False
        if not sites and use_contribution_sites:
            sites = fallback_sites
            self._merge_skipped(skipped, fallback_skipped)
        self._apply_site_weights(sites, site_weights)

        if not aggregated_metrics and not sites and not skipped:
            return

        aggregation = self._sanitize_json_object(info.get("aggregation"))
        key_metric = self._normalize_key_metric(info.get("key_metric"))
        record = {
            "round": current_round,
            "aggregated_metrics": aggregated_metrics,
            "sites": sites,
            "skipped_metrics": skipped,
        }
        if aggregation:
            record["aggregation"] = aggregation
        if key_metric:
            record["key_metric"] = key_metric
        self._append_round_record(fl_ctx, record)

        self._has_metrics = True
        self._final_round = current_round
        self._final_aggregated_metrics = aggregated_metrics
        self._metric_source = self._safe_text(info.get("metric_source"))
        self._metric_split = self._safe_text(info.get("metric_split"))
        self._aggregation = aggregation if aggregation else self._aggregation
        if key_metric:
            self._key_metric = key_metric

    @staticmethod
    def _to_fl_model(value):
        if isinstance(value, FLModel):
            return value
        if isinstance(value, Shareable):
            try:
                return FLModelUtils.from_shareable(value)
            except Exception:
                return None
        return None

    def _handle_global_best_model_available(self, fl_ctx: FLContext):
        selection = self._normalize_selection_info(fl_ctx.get_prop(AppConstants.METRICS_SELECTION_INFO, None))
        if not selection:
            return
        self._best_selection = selection
        key_metric = selection.get("key_metric")
        if key_metric:
            self._key_metric = key_metric

    def _handle_after_contribution_accept(self, fl_ctx: FLContext):
        accepted = fl_ctx.get_prop(AppConstants.AGGREGATION_ACCEPTED, True)
        if accepted is False:
            return

        result = fl_ctx.get_prop(AppConstants.TRAINING_RESULT, None)
        if result is None:
            peer_ctx = fl_ctx.get_peer_context()
            if peer_ctx:
                result = peer_ctx.get_prop(FLContextKey.SHAREABLE, None)
        if result is None:
            return

        try:
            model = FLModelUtils.from_shareable(result)
        except Exception:
            return
        if not model.metrics:
            return

        current_round = self._get_current_round(model, fl_ctx)
        skipped = []
        site_name = self._get_site_name(model, fl_ctx)
        metrics = self._normalize_metrics(model.metrics, site=site_name, skipped=skipped)
        if skipped:
            self._extend_round_skipped(current_round, skipped)
        if not metrics:
            return
        sites = self._round_sites.setdefault(current_round, [])
        if len(sites) >= self.max_sites_per_round:
            too_many_sites = []
            self._add_skipped(too_many_sites, site_name, "", "too_many_sites")
            self._extend_round_skipped(current_round, too_many_sites)
            return
        metric_count = self._round_site_metric_counts.get(current_round, 0)
        if metric_count + len(metrics) > self.max_site_metric_records_per_round:
            too_many_metrics = []
            self._add_skipped(too_many_metrics, site_name, "", "too_many_metrics")
            self._extend_round_skipped(current_round, too_many_metrics)
            allowed = self.max_site_metric_records_per_round - metric_count
            if allowed <= 0:
                return
            metrics = metrics[:allowed]
        self._round_site_metric_counts[current_round] = metric_count + len(metrics)
        site = {
            "name": self._sanitize_name(site_name),
            "metrics": metrics,
        }
        meta = model.meta or {}
        weight = self._safe_weight(meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND))
        if weight is not None:
            site["weight"] = weight
            site["weight_key"] = FLMetaKey.NUM_STEPS_CURRENT_ROUND
        sites.append(site)

    def _normalize_sites(self, sites, skipped):
        if not isinstance(sites, list):
            return []

        normalized_sites = []
        metric_count = 0
        for site in sites:
            if not isinstance(site, dict):
                continue
            if len(normalized_sites) >= self.max_sites_per_round:
                self._add_skipped(skipped, site.get("name"), "", "too_many_sites")
                break
            metrics = self._normalize_metrics(site.get("metrics"), site=site.get("name"), skipped=skipped)
            if not metrics:
                continue
            if metric_count + len(metrics) > self.max_site_metric_records_per_round:
                self._add_skipped(skipped, site.get("name"), "", "too_many_metrics")
                allowed = self.max_site_metric_records_per_round - metric_count
                if allowed <= 0:
                    break
                metrics = metrics[:allowed]
            metric_count += len(metrics)
            site_record = {
                "name": self._sanitize_name(site.get("name", "")),
                "metrics": metrics,
            }
            weight = self._safe_weight(site.get("weight"))
            if weight is not None:
                site_record["weight"] = weight
            weight_key = self._safe_text(site.get("weight_key"))
            if weight_key:
                site_record["weight_key"] = weight_key
            normalized_sites.append(site_record)
        return normalized_sites

    def _normalize_site_weights(self, site_weights, skipped):
        if not isinstance(site_weights, list):
            return {}

        normalized_weights = {}
        for site_weight in site_weights:
            if not isinstance(site_weight, dict):
                continue
            if len(normalized_weights) >= self.max_sites_per_round:
                self._add_skipped(skipped, site_weight.get("name"), "", "too_many_sites")
                break
            name = self._sanitize_name(site_weight.get("name", ""))
            if not name:
                continue
            record = {}
            weight = self._safe_weight(site_weight.get("weight"))
            if weight is not None:
                record["weight"] = weight
            weight_key = self._safe_text(site_weight.get("weight_key"))
            if weight_key:
                record["weight_key"] = weight_key
            if record:
                normalized_weights[name] = record
        return normalized_weights

    @staticmethod
    def _apply_site_weights(sites, site_weights):
        if not site_weights:
            return
        for site in sites:
            weight_info = site_weights.get(site.get("name"))
            if weight_info:
                site.update(weight_info)

    def _normalize_metrics(self, metrics, site, skipped, for_aggregation=False):
        if not isinstance(metrics, dict):
            return []

        result = []
        for name, value in metrics.items():
            if len(result) >= self.max_metrics_per_site_per_round:
                self._add_skipped(skipped, site, name, "too_many_metrics")
                break

            if not isinstance(name, str):
                self._add_skipped(skipped, site, "", "non_string_metric_name")
                continue
            metric_name = self._sanitize_name(name)
            if not metric_name:
                self._add_skipped(skipped, site, "", "empty_metric_name")
                continue
            normalized, reason = self._normalize_metric_value(value, for_aggregation=for_aggregation)
            if reason:
                self._add_skipped(skipped, site, metric_name, reason)
                continue
            result.append({"name": metric_name, "value": normalized})
        return result

    def _normalize_metric_value(self, value, for_aggregation=False) -> Tuple[Any, Optional[str]]:
        value = self._to_python_scalar(value)
        if isinstance(value, bool):
            return value, None
        if isinstance(value, int) and not isinstance(value, bool):
            if value.bit_length() > self.max_int_bit_length:
                return None, "number_too_large"
            return value, None
        if isinstance(value, float):
            if not math.isfinite(value):
                return None, "non_finite_number"
            return value, None
        if not for_aggregation and isinstance(value, str):
            if len(value) > self.max_string_value_length:
                return None, "string_too_long"
            return self._strip_control_chars(value), None
        return None, "unsupported_type"

    def _normalize_key_metric(self, key_metric):
        if not isinstance(key_metric, dict):
            return None
        name = key_metric.get("name")
        if not isinstance(name, str) or not name:
            return None
        normalized = {"name": self._sanitize_name(name)}
        mode = key_metric.get("mode")
        if mode in ("max", "min"):
            normalized["mode"] = mode
        mode_source = self._safe_text(key_metric.get("mode_source"))
        if mode_source:
            normalized["mode_source"] = mode_source
        return normalized

    def _normalize_selection_info(self, selection):
        if not isinstance(selection, dict):
            return None
        normalized = {}
        best_round = self._safe_round(selection.get("best_round"))
        if best_round is not None:
            normalized["best_round"] = best_round
        best_metrics = self._normalize_metric_collection(selection.get("best_metrics"), for_aggregation=True)
        if best_metrics:
            normalized["best_metrics"] = best_metrics
        best_aggregated_metrics = self._normalize_metric_collection(
            selection.get("best_aggregated_metrics"), for_aggregation=True
        )
        if best_aggregated_metrics:
            normalized["best_aggregated_metrics"] = best_aggregated_metrics
        key_metric = self._normalize_key_metric(selection.get("key_metric"))
        if key_metric:
            normalized["key_metric"] = key_metric
        source = self._safe_text(selection.get("source"))
        if source:
            normalized["best_metric_source"] = source
        metric_source = self._safe_text(selection.get("metric_source"))
        if metric_source:
            normalized["best_metric_detail_source"] = metric_source
        return normalized or None

    def _normalize_metric_collection(self, metrics, for_aggregation=False):
        if isinstance(metrics, dict):
            return self._normalize_metrics(metrics, site=None, skipped=[], for_aggregation=for_aggregation)
        if not isinstance(metrics, list):
            return []
        result = []
        for metric in metrics:
            if len(result) >= self.max_metrics_per_site_per_round:
                break
            if not isinstance(metric, dict):
                continue
            name = metric.get("name")
            if not isinstance(name, str):
                continue
            metric_name = self._sanitize_name(name)
            if not metric_name:
                continue
            normalized, reason = self._normalize_metric_value(metric.get("value"), for_aggregation=for_aggregation)
            if reason:
                continue
            result.append({"name": metric_name, "value": normalized})
        return result

    def _append_round_record(self, fl_ctx, record):
        self._ensure_paths(fl_ctx)
        os.makedirs(os.path.dirname(self._round_file_path), exist_ok=True)
        record = self._fit_round_record(record)
        line = self._safe_json_dumps(record)
        if line is None:
            record = self._make_minimal_round_record(record, "serialization_failed")
            line = self._safe_json_dumps(record)
        if line is None:
            return
        if len(line.encode("utf-8")) > self.max_round_record_bytes:
            record = self._make_minimal_round_record(record, "round_record_too_large")
            line = self._safe_json_dumps(record)
        if line is None or len(line.encode("utf-8")) > self.max_round_record_bytes:
            return
        with open(self._round_file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _write_summary_if_needed(self, fl_ctx):
        if not self._has_metrics:
            return
        self._ensure_paths(fl_ctx)
        summary = {
            "schema_version": "1",
            "status": "metrics_reported",
            "final_round": self._final_round,
            "final_aggregated_metrics": self._final_aggregated_metrics,
            "round_metrics_file": self.round_file_name,
            "notes": [
                "Aggregated metrics are weighted averages of client-reported metric values.",
                "Nonlinear metrics are not recomputed from pooled predictions.",
            ],
        }
        job_name = self._safe_text(fl_ctx.get_job_id())
        if job_name:
            summary["job_name"] = job_name
        if self._metric_source:
            summary["metric_source"] = self._metric_source
        if self._metric_split:
            summary["metric_split"] = self._metric_split
        if self._best_selection and self._best_selection.get("key_metric"):
            summary["key_metric"] = self._best_selection["key_metric"]
        elif self._key_metric:
            summary["key_metric"] = self._key_metric
        if self._best_selection:
            for field in (
                "best_round",
                "best_metrics",
                "best_aggregated_metrics",
                "best_metric_source",
                "best_metric_detail_source",
            ):
                if field in self._best_selection:
                    summary[field] = self._best_selection[field]
        if self._aggregation:
            summary["aggregation"] = self._sanitize_json_object(self._aggregation)

        data = self._safe_json_dumps(summary, indent=2)
        if data is None:
            return
        if len(data.encode("utf-8")) > self.max_summary_bytes:
            summary.pop("notes", None)
            data = self._safe_json_dumps(summary, indent=2)
        if data is None or len(data.encode("utf-8")) > self.max_summary_bytes:
            return
        os.makedirs(os.path.dirname(self._summary_file_path), exist_ok=True)
        with open(self._summary_file_path, "w", encoding="utf-8") as f:
            f.write(data)

    def _fit_round_record(self, record):
        fitted = {
            "round": record.get("round"),
            "aggregated_metrics": self._fit_json_list(
                record.get("aggregated_metrics", []), self.max_round_record_bytes
            ),
            "sites": [],
            "skipped_metrics": [],
        }
        for field in ("aggregation", "key_metric"):
            if field in record:
                fitted[field] = record[field]
        used = len((self._safe_json_dumps(fitted) or "").encode("utf-8"))
        for field in ("sites", "skipped_metrics"):
            for item in record.get(field, []):
                item_json = self._safe_json_dumps(item)
                if item_json is None:
                    continue
                next_size = used + len(item_json.encode("utf-8")) + 2
                if next_size > self.max_round_record_bytes:
                    fitted["skipped_metrics"].append({"name": "", "reason": f"{field}_truncated"})
                    return fitted
                fitted[field].append(item)
                used = next_size
        return fitted

    def _fit_json_list(self, items, max_bytes):
        result = []
        used = 0
        for item in items:
            item_json = self._safe_json_dumps(item)
            if item_json is None:
                continue
            item_size = len(item_json.encode("utf-8")) + 1
            if used + item_size > max_bytes:
                break
            result.append(item)
            used += item_size
        return result

    def _make_minimal_round_record(self, record, reason):
        return {
            "round": record.get("round"),
            "aggregated_metrics": self._fit_json_list(
                record.get("aggregated_metrics", []), self.max_round_record_bytes
            ),
            "sites": [],
            "skipped_metrics": [{"name": "", "reason": reason}],
        }

    @staticmethod
    def _safe_json_dumps(value, indent=None):
        try:
            return json.dumps(value, allow_nan=False, indent=indent, separators=None if indent else (",", ":"))
        except (TypeError, ValueError, OverflowError):
            return None

    def _ensure_paths(self, fl_ctx):
        if self._summary_file_path and self._round_file_path:
            return
        self._validate_file_name(self.summary_file_name, "metrics summary file name")
        self._validate_file_name(self.round_file_name, "round metrics file name")
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_job_id())
        self._summary_file_path = resolve_path_under_root(
            run_dir, os.path.join(self.results_dir, self.summary_file_name), "metrics summary path"
        )
        self._round_file_path = resolve_path_under_root(
            run_dir, os.path.join(self.results_dir, self.round_file_name), "round metrics path"
        )

    @staticmethod
    def _validate_file_name(file_name: str, path_name: str):
        if not isinstance(file_name, str):
            raise TypeError(f"{path_name} must be str but got {type(file_name)}")
        if os.path.basename(file_name) != file_name or file_name in ("", ".", ".."):
            raise ValueError(f"{path_name} {file_name} must be relative and stay inside the metrics directory.")

    def _get_current_round(self, aggr_result, fl_ctx):
        if aggr_result.current_round is not None:
            return aggr_result.current_round
        meta = aggr_result.meta or {}
        for key in (AppConstants.CURRENT_ROUND, "current_round"):
            value = meta.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        value = fl_ctx.get_prop(AppConstants.CURRENT_ROUND, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    def _get_site_name(self, model, fl_ctx):
        meta = model.meta or {}
        for key in (FLMetaKey.SITE_NAME, "client_name", "site_name"):
            value = meta.get(key)
            if isinstance(value, str) and value:
                return value
        peer_ctx = fl_ctx.get_peer_context()
        if peer_ctx:
            identity = peer_ctx.get_identity_name(default=None)
            if identity:
                return identity
        return AppConstants.CLIENT_UNKNOWN

    @staticmethod
    def _safe_round(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        return None

    def _safe_number(self, value):
        value = self._to_python_scalar(value)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            if value.bit_length() <= self.max_int_bit_length:
                return value
            return None
        if isinstance(value, float) and math.isfinite(value):
            return value
        return None

    def _safe_weight(self, value):
        weight = self._safe_number(value)
        if weight is None or weight <= 0:
            return None
        return weight

    @staticmethod
    def _to_python_scalar(value):
        if isinstance(value, np.generic):
            try:
                return value.item()
            except (TypeError, ValueError, OverflowError):
                return value
        return value

    def _safe_text(self, value):
        if not isinstance(value, str):
            return None
        return self._strip_control_chars(value[: self.max_metric_name_length])

    def _sanitize_name(self, name):
        if not isinstance(name, str):
            return ""
        return self._strip_control_chars(name[: self.max_metric_name_length])

    @staticmethod
    def _strip_control_chars(value: str):
        return "".join(ch for ch in value if ord(ch) >= 32 and ch != "\x7f")

    def _add_skipped(self, skipped, site, name, reason):
        if len(skipped) >= self.max_skipped_metrics_per_round:
            return
        record = {"name": self._sanitize_name(name), "reason": reason}
        if site is not None:
            record["site"] = self._sanitize_name(site)
        skipped.append(record)

    def _extend_round_skipped(self, current_round, skipped):
        if not skipped:
            return
        round_skipped = self._round_skipped.setdefault(current_round, [])
        self._merge_skipped(round_skipped, skipped)

    def _merge_skipped(self, target, skipped):
        for record in skipped:
            if len(target) >= self.max_skipped_metrics_per_round:
                return
            target.append(record)

    def _sanitize_json_object(self, value):
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                if not isinstance(key, str):
                    continue
                child_value = self._sanitize_json_object(child)
                if child_value is not None:
                    result[self._sanitize_name(key)] = child_value
            return result
        if isinstance(value, list):
            return [v for v in (self._sanitize_json_object(child) for child in value) if v is not None]
        normalized, reason = self._normalize_metric_value(value, for_aggregation=False)
        if reason:
            return None
        return normalized

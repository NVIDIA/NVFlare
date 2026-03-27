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

import time
from typing import Dict

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_event_type import AppEventType
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.metrics.metrics_keys import MetricKeys, MetricTypes
from nvflare.metrics.metrics_publisher import collect_metrics


class QuantumProofMetricsCollector(FLComponent):
    """Emit proof-path and PQC posture metrics for observability dashboards.

    This collector is intentionally lightweight and does not perform cryptographic
    verification by itself. It provides operational metrics for the "path to
    quantum proofs" by tracking proof-related lifecycle events and runtime posture.
    """

    def __init__(
        self,
        tags: Dict,
        streaming_to_server: bool = False,
        expected_attestation_mode: str = "xmss",
        expected_kex_mode: str = "x25519-mlkem768-hybrid",
        pqc_migration_enabled: bool = True,
        pqc_lock_legacy_transfers: bool = True,
    ):
        super().__init__()
        self.tags = tags or {}
        self.streaming_to_server = streaming_to_server
        self.expected_attestation_mode = expected_attestation_mode
        self.expected_kex_mode = expected_kex_mode
        self.pqc_migration_enabled = pqc_migration_enabled
        self.pqc_lock_legacy_transfers = pqc_lock_legacy_transfers
        self.data_bus = DataBus()

        self._proof_verify_start_time = None
        self._aggregation_start_time = None

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._publish_runtime_posture(fl_ctx)
            return

        if event == EventType.BEFORE_TASK_EXECUTION:
            self._proof_verify_start_time = time.time()
            self._publish_counter("quantum_proof_verify", fl_ctx)
            return

        if event == EventType.AFTER_TASK_EXECUTION:
            self._publish_counter("quantum_proof_verify_success", fl_ctx)
            self._publish_elapsed(self._proof_verify_start_time, "quantum_proof_verify", fl_ctx)
            self._proof_verify_start_time = None
            return

        if event == EventType.ABORT_TASK:
            self._publish_counter("quantum_proof_verify_failure", fl_ctx)
            self._proof_verify_start_time = None
            return

        if event == AppEventType.BEFORE_AGGREGATION:
            self._aggregation_start_time = time.time()
            self._publish_counter("quantum_proof_aggregation", fl_ctx)
            return

        if event == AppEventType.END_AGGREGATION:
            self._publish_counter("quantum_proof_aggregation_success", fl_ctx)
            self._publish_elapsed(self._aggregation_start_time, "quantum_proof_aggregation", fl_ctx)
            self._aggregation_start_time = None

    def _publish_runtime_posture(self, fl_ctx: FLContext):
        tags = self._get_tags(fl_ctx)
        posture_tags = {
            **tags,
            "attestation_mode": self.expected_attestation_mode,
            "kex_mode": self.expected_kex_mode,
        }

        self._publish_metric(
            metric_name="quantum_path",
            metrics={"ready": 1, MetricKeys.type: MetricTypes.GAUGE},
            tags=posture_tags,
            fl_ctx=fl_ctx,
        )

        self._publish_metric(
            metric_name="quantum_pqc_controls",
            metrics={
                "migration_enabled": 1 if self.pqc_migration_enabled else 0,
                "legacy_lock_enabled": 1 if self.pqc_lock_legacy_transfers else 0,
                MetricKeys.type: MetricTypes.GAUGE,
            },
            tags=posture_tags,
            fl_ctx=fl_ctx,
        )

    def _publish_counter(self, metric_name: str, fl_ctx: FLContext):
        self._publish_metric(
            metric_name=metric_name,
            metrics={MetricKeys.count: 1, MetricKeys.type: MetricTypes.COUNTER},
            tags=self._get_tags(fl_ctx),
            fl_ctx=fl_ctx,
        )

    def _publish_elapsed(self, start_time, metric_name: str, fl_ctx: FLContext):
        if start_time is None:
            return

        elapsed = max(time.time() - start_time, 0.0)
        self._publish_metric(
            metric_name=metric_name,
            metrics={MetricKeys.time_taken: elapsed, MetricKeys.type: MetricTypes.GAUGE},
            tags=self._get_tags(fl_ctx),
            fl_ctx=fl_ctx,
        )

    def _publish_metric(self, metric_name: str, metrics: dict, tags: dict, fl_ctx: FLContext):
        collect_metrics(
            self,
            self.streaming_to_server,
            metrics,
            metric_name,
            tags,
            self.data_bus,
            fl_ctx,
        )

    def _get_tags(self, fl_ctx: FLContext) -> dict:
        tags = dict(self.tags)
        job_id = fl_ctx.get_job_id()
        if job_id:
            tags["job_id"] = job_id
        return tags

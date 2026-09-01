import time

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_common.workflows.fedavg import center_message
from nvflare.app_common.utils.fedprox_utils import set_fedprox_metadata
from nvflare.app_common.utils.tensor_disk_offload_context import (
    cleanup_tensor_disk_offload,
    setup_tensor_disk_offload,
)
from nvflare.app_common.app_event_type import AppEventType


class FedSCS(FedAvg):
    """FedSCS controller.

    Reuses FedAvg orchestration while using FedSCSAggregator
    for model aggregation.
    """

    def run(self) -> None:
        disk_offload_context = None

        try:
            disk_offload_context = setup_tensor_disk_offload(
                engine=getattr(self, "engine", None),
                enabled=self.enable_tensor_disk_offload,
                job_id=self.fl_ctx.get_job_id("job"),
            )

            if self.enable_tensor_disk_offload and not disk_offload_context.applied:
                self.warning(
                    "enable_tensor_disk_offload=True but no active cell is available; "
                    "falling back to in-memory tensor download"
                )

            self.info(center_message("Start FedSCS."))

            self.fl_ctx.set_prop(
                AppConstants.NUM_ROUNDS,
                self.num_rounds,
                private=True,
                sticky=False,
            )

            if self.model is not None:
                if isinstance(self.model, FLModel):
                    model = self.model
                else:
                    model = FLModel(params=self.model)
                self.info("Using provided model")
            else:
                model = self.load_model()

            model.start_round = self.start_round
            model.total_rounds = self.num_rounds

            for self.current_round in range(
                self.start_round,
                self.start_round + self.num_rounds,
            ):
                self.info(
                    center_message(
                        f"FedSCS Round {self.current_round} started.",
                        boarder_str="-",
                    )
                )

                model.current_round = self.current_round

                self.fl_ctx.set_prop(
                    AppConstants.CURRENT_ROUND,
                    self.current_round,
                    private=True,
                    sticky=False,
                )

                if self.aggregator and self.aggregator.fl_ctx:
                    self.aggregator.fl_ctx.set_prop(
                        AppConstants.CURRENT_ROUND,
                        self.current_round,
                        private=True,
                        sticky=False,
                    )

                self.event(AppEventType.ROUND_STARTED)

                clients = self.sample_clients(self.num_clients)

                if self.aggregator:
                    self.aggregator.reset_stats()
                else:
                    self._aggr_helper = WeightedAggregationHelper(
                        exclude_vars=self.exclude_vars
                    )
                    self._aggr_metrics_helper = WeightedAggregationHelper()
                    self._all_metrics = True

                self._received_count = 0
                self._expected_count = len(clients)
                self._params_type = None
                self._site_metric_weights = {}

                set_fedprox_metadata(model, self.fedprox_mu)

                self.send_model(
                    task_name=self.task_name,
                    targets=clients,
                    data=model,
                    callback=self._aggregate_one_result,
                )

                while self.get_num_standing_tasks():
                    if self.abort_signal.triggered:
                        self.info("Abort signal triggered. Finishing FedSCS.")
                        return
                    time.sleep(self._task_check_period)

                self.event(AppEventType.BEFORE_AGGREGATION)

                aggregate_results = self._get_aggregated_result()

                self.fire_event_with_data(
                    AppEventType.AFTER_AGGREGATION,
                    self.fl_ctx,
                    AppConstants.AGGREGATION_RESULT,
                    aggregate_results,
                )

                model = self.update_model(model, aggregate_results)

                if self.stop_condition:
                    self.info(
                        f"Round {self.current_round} global metrics: {model.metrics}"
                    )

                    if self.is_curr_model_better(model):
                        self.info("New best model found.")
                        self.save_model(model)
                    elif self.patience:
                        self.info(
                            "No metric improvement, "
                            f"num of FL rounds without improvement: "
                            f"{self.num_fl_rounds_without_improvement}"
                        )

                    if self.should_stop(model.metrics):
                        self.info(
                            f"Stopping at round={self.current_round} "
                            f"out of total_rounds={self.num_rounds}."
                        )
                        break
                else:
                    self.save_model(model)

                self._maybe_cleanup_memory()

            self.info(center_message("Finished FedSCS."))

        finally:
            cleanup_tensor_disk_offload(
                engine=getattr(self, "engine", None),
                context=disk_offload_context,
            )

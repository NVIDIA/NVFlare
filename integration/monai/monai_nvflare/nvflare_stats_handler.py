# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch
from monai.config import IgniteInfo
from monai.utils import is_scalar, min_version, optional_import

from nvflare.apis.analytix import AnalyticsDataType, LogWriterName
from nvflare.app_common.tracking.log_writer import LogWriter

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")

if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import(
        "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine", as_type="decorator"
    )


ANALYTIC_EVENT_TYPE = "analytix_log_stats"
DEFAULT_TAG = "Loss"


class NVFlareStatsHandler(LogWriter):
    """
    NVFlareStatsHandler defines a set of Ignite Event-handlers for all the NVFlare ``LogWriter`` logics.
    It can be used for any Ignite Engine(trainer, validator and evaluator).
    And it can support both epoch level and iteration level with pre-defined LogWriter event sender.
    The expected data source is Ignite ``engine.state.output`` and ``engine.state.metrics``.

    Default behaviors:
        - When EPOCH_COMPLETED, write each dictionary item in
          ``engine.state.metrics`` to TensorBoard.
        - When ITERATION_COMPLETED, write each dictionary item in
          ``self.output_transform(engine.state.output)`` to TensorBoard.

    """

    def __init__(
        self,
        iteration_log: bool | Callable[[Engine, int], bool] = True,
        epoch_log: bool | Callable[[Engine, int], bool] = True,
        output_transform: Callable = lambda x: x[0],
        global_epoch_transform: Callable = lambda x: x,
        state_attributes: Sequence[str] | None = None,
        state_attributes_type: AnalyticsDataType | None = None,
        tag_name: str = DEFAULT_TAG,
        metrics_sender_id: str = None,
    ) -> None:
        """
        Args:
            iteration_log: whether to send data when iteration completed, default to `True`.
                ``iteration_log`` can be also a function and it will be interpreted as an event filter
                (see https://pytorch.org/ignite/generated/ignite.engine.events.Events.html for details).
                Event filter function accepts as input engine and event value (iteration) and should return True/False.
            epoch_log: whether to send data when epoch completed, default to `True`.
                ``epoch_log`` can be also a function and it will be interpreted as an event filter.
                See ``iteration_log`` argument for more details.
            output_transform: a callable that is used to transform the
                ``ignite.engine.state.output`` into a scalar to plot, or a dictionary of {key: scalar}.
                In the latter case, the output string will be formatted as key: value.
                By default this value plotting happens when every iteration completed.
                The default behavior is to print loss from output[0] as output is a decollated list
                and we replicated loss value for every item of the decollated list.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            global_epoch_transform: a callable that is used to customize global epoch number.
                For example, in evaluation, the evaluator engine might want to use trainer engines epoch number
                when plotting epoch vs metric curves.
            state_attributes: expected attributes from `engine.state`, if provided, will extract them
                when epoch completed.
            state_attributes_type: the type of the expected attributes from `engine.state`.
                Only required when `state_attributes` is not None.
            tag_name: when iteration output is a scalar, tag_name is used to plot, defaults to ``'Loss'``.
            metrics_sender_id (str): provided for LogWriter to get MetricsExchanger
        """

        super().__init__(metrics_sender_id=metrics_sender_id)
        self.iteration_log = iteration_log
        self.epoch_log = epoch_log
        self.output_transform = output_transform
        self.global_epoch_transform = global_epoch_transform
        self.state_attributes = state_attributes
        self.state_attributes_type = state_attributes_type
        self.tag_name = tag_name

    def attach(self, engine: Engine) -> None:
        """
        Register a set of Ignite Event-Handlers to a specified Ignite engine.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.iteration_log and not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            event = Events.ITERATION_COMPLETED
            if callable(self.iteration_log):  # substitute event with new one using filter callable
                event = event(event_filter=self.iteration_log)
            engine.add_event_handler(event, self.iteration_completed)
        if self.epoch_log and not engine.has_event_handler(self.epoch_completed, Events.EPOCH_COMPLETED):
            event = Events.EPOCH_COMPLETED
            if callable(self.epoch_log):  # substitute event with new one using filter callable
                event = event(event_filter=self.epoch_log)
            engine.add_event_handler(event, self.epoch_completed)

    def epoch_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation epoch completed Event.
        Write epoch level events, default values are from Ignite `engine.state.metrics` dict.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        self._default_epoch_sender(engine)

    def iteration_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation iteration completed Event.
        Write iteration level events, default values are from Ignite `engine.state.output`.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        self._default_iteration_sender(engine)

    def _send_stats(self, _engine: Engine, tag: str, value: Any, data_type: AnalyticsDataType, step: int) -> None:
        """
        Write value.

        Args:
            _engine: Ignite Engine, unused argument.
            tag: tag name in the TensorBoard.
            value: value of the scalar data for current step.
            step: index of current step.

        """
        self.sender.add(tag=tag, value=value, data_type=data_type, global_step=step)

    def _default_epoch_sender(self, engine: Engine) -> None:
        """
        Execute epoch level event write operation.
        Default to write the values from Ignite `engine.state.metrics` dict and
        write the values of specified attributes of `engine.state`.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        current_epoch = self.global_epoch_transform(engine.state.epoch)
        summary_dict = engine.state.metrics
        for name, value in summary_dict.items():
            self._send_stats(engine, name, value, AnalyticsDataType.SCALAR, current_epoch)

        if self.state_attributes is not None:
            for attr in self.state_attributes:
                self._send_stats(
                    engine, attr, getattr(engine.state, attr, None), self.state_attributes_type, current_epoch
                )

    def _default_iteration_sender(self, engine: Engine) -> None:
        """
        Execute iteration level event write operation based on Ignite `engine.state.output` data.
        Extract the values from `self.output_transform(engine.state.output)`.
        Since `engine.state.output` is a decollated list and we replicated the loss value for every item
        of the decollated list, the default behavior is to track the loss from `output[0]`.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        loss = self.output_transform(engine.state.output)
        if loss is None:
            return  # do nothing if output is empty
        if isinstance(loss, dict):
            for name in sorted(loss):
                value = loss[name]
                if not is_scalar(value):
                    warnings.warn(
                        "ignoring non-scalar output in NVFlareStatsHandler,"
                        " make sure `output_transform(engine.state.output)` returns"
                        " a scalar or dictionary of key and scalar pairs to avoid this warning."
                        " {}:{}".format(name, type(value))
                    )
                    continue  # not plot multi dimensional output
                self._send_stats(
                    _engine=engine,
                    tag=name,
                    data_type=AnalyticsDataType.SCALAR,
                    value=value.item() if isinstance(value, torch.Tensor) else value,
                    step=engine.state.iteration,
                )
        elif is_scalar(loss):  # not printing multi dimensional output
            self._send_stats(
                _engine=engine,
                tag=self.tag_name,
                data_type=AnalyticsDataType.SCALAR,
                value=loss.item() if isinstance(loss, torch.Tensor) else loss,
                step=engine.state.iteration,
            )
        else:
            warnings.warn(
                "ignoring non-scalar output in NVFlareStatsHandler,"
                " make sure `output_transform(engine.state.output)` returns"
                " a scalar or a dictionary of key and scalar pairs to avoid this warning."
                " {}".format(type(loss))
            )

    def get_writer_name(self) -> LogWriterName:
        """Not used, just for abstractmethod"""
        return LogWriterName.MLFLOW

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
import copy
import time
from abc import ABC, abstractmethod
from typing import Any

from nvflare.apis.signal import Signal

from .config import process_train_config


class Context(dict):

    def fire_event(self, event_type: str, data: Any, abort_signal: Signal):
        handlers = self.get(ContextKey.EVENT_HANDLERS)
        if handlers:
            for h in handlers:
                h.handle_event(event_type, data, self, abort_signal)


class ContextKey:
    RUNNER = "runner"
    PARAMS = "params"
    TRAINER = "trainer"
    COMPONENTS = "components"
    EVENT_HANDLERS = "event_handlers"
    FILTERS = "filters"


class ComponentKey:
    TRAINER = "trainer"
    EVENT_HANDLERS = "event_handlers"
    FILTERS = "filters"


class EventType:
    BEFORE_TRAIN = "before_train"
    AFTER_TRAIN = "after_train"
    LOSS_GENERATED = "loss_generated"


class Batch:
    pass


class Dataset(ABC):

    def get_next_batch(self) -> Batch:
        pass


class Model:
    pass


class DataSource(ABC):

    @abstractmethod
    def get_dataset(self, dataset_type: str, ctx: Context) -> Dataset:
        pass


class Evaluator(ABC):

    @abstractmethod
    def evaluate(self, model: Model, dataset: Dataset, ctx: Context, abort_signal: Signal) -> Any:
        pass


class Trainer(ABC):

    @abstractmethod
    def train(self, data_source: DataSource, model: Model, ctx: Context, abort_signal: Signal) -> Model:
        pass


class Filter(ABC):

    @abstractmethod
    def filter(self, model: Model, ctx: Context, abort_signal: Signal) -> Model:
        pass


class Transform(ABC):

    @abstractmethod
    def transform(self, batch: Batch, ctx: Context, abort_signal: Signal) -> Batch:
        pass


class EventHandler(ABC):

    @abstractmethod
    def handle_event(self, event_type: str, event_data: Any, ctx: Context, abort_signal: Signal):
        pass


class FlareRunner:

    def __init__(
        self,
        data_source: DataSource,
        device_info: dict,
        user_info: dict,
        job_timeout: float,
        creator_registry: dict = None,
    ):
        if not creator_registry:
            creator_registry = {}
        self.data_source = data_source
        self.creator_registry = creator_registry
        self.device_info = device_info
        self.user_info = user_info
        self.job_timeout = job_timeout
        self.abort_signal = Signal()
        self.job_id = None
        self.job_name = None
        self.cookie = None

        # add built-in creators
        self._add_builtin_creators()

    def _add_builtin_creators(self):
        """Add creators for Flare's builtin components

        Returns:

        """
        pass

    def run(self):
        while True:
            sess_done = self._do_one_job()
            if sess_done:
                return

    def stop(self):
        if self.abort_signal:
            self.abort_signal.trigger(True)

    def _get_job(self, ctx: Context, abort_signal: Signal) -> Any:
        """Repeatedly try to get job from host

        Returns: a job or None if the host says DONE or timed out.

        """
        pass

    def _get_task(self, ctx: Context, abort_signal: Signal) -> (Any, bool):
        """Repeatedly try to get a task from the host

        Returns: a task or None if the host says DONE

        """
        pass

    def _report_result(self, result: Model, ctx: Context, abort_signal: Signal) -> bool:
        pass

    def _do_one_job(self) -> bool:
        """Work with the host to do one job

        Returns: whether whole session is done.

        """
        ctx = Context()
        ctx[ContextKey.RUNNER] = self

        # try to get job
        job = self._get_job(ctx, self.abort_signal)
        if not job:
            # No job for me.
            return True

        self.job_name = job.name
        self.job_id = job.job_id

        # the job_data in the job contains trainer config!
        components, filters, handlers = process_train_config(job.job_data, self.creator_registry)
        trainer = components.get(ComponentKey.TRAINER)
        if not trainer:
            raise RuntimeError("bad trainer config: no trainer")
        assert isinstance(trainer, Trainer)

        ctx[ContextKey.COMPONENTS] = components
        ctx[ContextKey.EVENT_HANDLERS] = handlers
        ctx[ContextKey.FILTERS] = filters

        while True:
            task, sess_done = self._get_task(ctx, self.abort_signal)
            if self.abort_signal.triggered:
                return True

            if not task:
                # no more work for this job
                return sess_done

            # create a new context for each task!
            task_ctx = copy.copy(ctx)

            self.cookie = task.cooke
            model = task.model
            params = task.params
            task_ctx[ContextKey.PARAMS] = params

            task_ctx.fire_event(EventType.BEFORE_TRAIN, time.time(), self.abort_signal)
            trained_model = trainer.train(self.data_source, model, task_ctx, self.abort_signal)
            task_ctx.fire_event(EventType.AFTER_TRAIN, (time.time(), trained_model), self.abort_signal)

            if self.abort_signal.triggered:
                return True

            # filter the output if needed
            if filters:
                for f in filters:
                    assert isinstance(f, Filter)
                    trained_model = f.filter(trained_model, task_ctx, self.abort_signal)
                    if self.abort_signal.triggered:
                        return True

            sess_done = self._report_result(trained_model, task_ctx, self.abort_signal)
            if sess_done:
                return sess_done

            if self.abort_signal.triggered:
                return True

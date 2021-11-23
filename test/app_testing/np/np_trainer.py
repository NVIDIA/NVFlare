import os
import time

import numpy as np

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.app_constant import AppConstants

from .constants import NPConstants


class NPTrainer(Executor):
    def __init__(
        self,
        delta=1,
        sleep_time=0,
        train_task_name=AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        model_name="best_numpy.npy",
        model_dir="model",
    ):
        # Init functions of components should be very minimal. Init
        # is called when json is read. A big init will cause json loading to halt
        # for long time.
        super().__init__()

        if not (isinstance(delta, float) or isinstance(delta, int)):
            raise TypeError("delta must be an instance of float or int.")

        self._delta = delta
        self._model_name = model_name
        self._model_dir = model_dir
        self._sleep_time = sleep_time
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        # if event_type == EventType.START_RUN:
        #     Create all major components here. This is a simple app that doesn't need any components.
        # elif event_type == EventType.END_RUN:
        #     # Clean up resources (closing files, joining threads, removing dirs etc)
        pass

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        # Any long tasks should check abort_signal regularly. Otherwise abort client
        # will not work.
        count, interval = 0, 0.5
        while count < self._sleep_time:
            if abort_signal.triggered:
                return self._get_exception_shareable()
            time.sleep(interval)
            count += interval

        if task_name == self._train_task_name:
            # First we extract DXO from the shareable.
            try:
                incoming_dxo = from_shareable(shareable)
            except BaseException as e:
                self.system_panic(f"Unable to convert shareable to model definition. Exception {e.__str__()}", fl_ctx)
                return self._get_exception_shareable()

            # Information about workflow is retreived from the shareable header.
            current_round = shareable.get_header(AppConstants.CURRENT_ROUND, None)
            total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS, None)

            # Ensure that data is of type weights. Extract model data.
            if incoming_dxo.data_kind != DataKind.WEIGHTS:
                self.system_panic("Model dex should be of kind DataKind.WEIGHTS.", fl_ctx)
                return self._get_exception_shareable()
            np_data = incoming_dxo.data

            # Display properties.
            self.log_info(fl_ctx, f"Incoming data kind: {incoming_dxo.data_kind}")
            self.log_info(fl_ctx, f"Model: \n{np_data}")
            self.log_info(fl_ctx, f"Current Round: {current_round}")
            self.log_info(fl_ctx, f"Total Rounds: {total_rounds}")
            self.log_info(fl_ctx, f"Task name: {task_name}")
            self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

            # Check abort signal
            if abort_signal.triggered:
                return self._get_exception_shareable()

            # Doing some dummy training.
            if np_data:
                if NPConstants.NUMPY_KEY in np_data:
                    np_data[NPConstants.NUMPY_KEY] += self._delta
                else:
                    self.log_error(fl_ctx, "numpy_key not found in model.")
                    shareable.set_return_code(ReturnCode.EXECUTION_RESULT_ERROR)
                    return shareable
            else:
                self.log_error(fl_ctx, "No model weights found in shareable.")
                shareable.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
                return shareable

            # We check abort_signal regularly to make sure
            if abort_signal.triggered:
                return self._get_exception_shareable()

            # Save local numpy model
            try:
                self._save_local_model(fl_ctx, np_data)
            except Exception as e:
                self.log_error(fl_ctx, f"Exception in saving local model: {e}.")

            self.log_info(
                fl_ctx,
                f"Model after training: {np_data}",
            )

            # Checking abort signal again.
            if abort_signal.triggered:
                return self._get_exception_shareable()

            # Prepare a DXO for our updated model. Create shareable and return
            outgoing_dxo = DXO(data_kind=incoming_dxo.data_kind, data=np_data, meta={})
            return outgoing_dxo.to_shareable()
        elif task_name == self._submit_model_task_name:
            # Retrieve the local model saved during training.
            np_data = None
            try:
                np_data = self._load_local_model(fl_ctx)
            except Exception as e:
                self.log_error(fl_ctx, f"Unable to load model: {e}")

            # Checking abort signal
            if abort_signal.triggered:
                return self._get_exception_shareable()

            # Create DXO and shareable from model data.
            model_shareable = Shareable()
            if np_data:
                outgoing_dxo = DXO(data_kind=DataKind.WEIGHTS, data=np_data)
                model_shareable = outgoing_dxo.to_shareable()
            else:
                # Set return code.
                self.log_error(fl_ctx, f"local model not found.")
                model_shareable.set_return_code(ReturnCode.EXECUTION_RESULT_ERROR)

            return model_shareable
        else:
            # If unknown task name, set RC accordingly.
            shareable = Shareable()
            shareable.set_return_code(ReturnCode.TASK_UNKNOWN)
            return shareable

    def _load_local_model(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        run_dir = engine.get_workspace().get_run_dir(run_number)
        model_path = os.path.join(run_dir, self._model_dir)

        model_load_path = os.path.join(model_path, self._model_name)
        np_data = None
        try:
            np_data = np.load(model_load_path)
        except Exception as e:
            self.log_error(fl_ctx, f"Unable to load local model: {e.__str__()}")
            return None

        model = ModelLearnable()
        model[NPConstants.NUMPY_KEY] = np_data

        return model

    def _save_local_model(self, fl_ctx: FLContext, model: dict):
        # Save local model
        engine = fl_ctx.get_engine()
        run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        run_dir = engine.get_workspace().get_run_dir(run_number)
        model_path = os.path.join(run_dir, self._model_dir)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_save_path = os.path.join(model_path, self._model_name)
        if model_save_path:
            with open(model_save_path, "wb") as f:
                np.save(f, model[NPConstants.NUMPY_KEY])
            self.log_info(fl_ctx, f"Saved numpy model to: {model_save_path}")

    def _get_exception_shareable(self) -> Shareable:
        """Abort execution. This is used if abort_signal is triggered. Users should
        make sure they abort any running processes here.

        Returns:
            Shareable: Shareable with return_code.
        """
        shareable = Shareable()
        shareable.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
        return shareable

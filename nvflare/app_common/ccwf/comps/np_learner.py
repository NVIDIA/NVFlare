import os
from typing import Union

import numpy as np

from nvflare.apis.dxo import MetaKey
from nvflare.apis.shareable import ReturnCode
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.abstract.model_learner import ModelLearner
from nvflare.app_common.np.constants import NPConstants


class NPLearner(ModelLearner):
    def __init__(self):
        ModelLearner.__init__(self)
        self._model_name = "best_numpy.npy"
        self._model_dir = "model"
        self._delta = 1

    def train(self, model: FLModel) -> Union[str, FLModel]:
        # Ensure that data is of type weights. Extract model data.
        if model.params_type != ParamsType.FULL:
            self.stop_task("Model DXO should be of kind DataKind.WEIGHTS.")
            return ReturnCode.BAD_TASK_DATA

        np_data = model.params

        # Display properties.
        self.info(f"Incoming data kind: {model.params_type}")
        self.info(f"Model: \n{np_data}")
        self.info(f"Current Round: {model.current_round}")
        self.info(f"Total Rounds: {model.total_rounds}")
        self.info(f"Client identity: {self.site_name}")

        # Check abort signal
        if self.is_aborted():
            return ReturnCode.TASK_ABORTED

        # Doing some dummy training.
        if np_data:
            if NPConstants.NUMPY_KEY in np_data:
                np_data[NPConstants.NUMPY_KEY] += self._delta
            else:
                self.error("numpy_key not found in model.")
                return ReturnCode.BAD_TASK_DATA
        else:
            self.error("No model weights found in task data.")
            return ReturnCode.BAD_TASK_DATA

        # We check abort_signal regularly to make sure
        if self.is_aborted():
            return ReturnCode.TASK_ABORTED

        # Save local numpy model
        try:
            self._save_local_model(np_data)
        except Exception as e:
            self.error(f"Exception in saving local model: {e}.")

        self.info(f"Model after training: {np_data}")

        # Checking abort signal again.
        if self.is_aborted():
            return ReturnCode.TASK_ABORTED

        # Prepare a DXO for our updated model. Create shareable and return
        return FLModel(params_type=ParamsType.FULL, params=np_data, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1})

    def _save_local_model(self, model: dict):
        # Save local model
        run_dir = self.app_root

        self.info(f"App Root: {self.app_root}")

        model_path = os.path.join(run_dir, self._model_dir)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_save_path = os.path.join(model_path, self._model_name)
        np.save(model_save_path, model[NPConstants.NUMPY_KEY])
        self.info(f"Saved numpy model to: {model_save_path}")

    def get_model(self, model_name: str) -> Union[str, FLModel]:
        # Retrieve the local model saved during training.
        np_data = None
        try:
            np_data = self._load_local_model()
        except Exception as e:
            self.error(f"Unable to load model: {e}")

        # Create DXO from model data.
        if np_data:
            return FLModel(params_type=ParamsType.FULL, params=np_data)
        else:
            return ReturnCode.EXECUTION_RESULT_ERROR

    def _load_local_model(self):
        run_dir = self.app_root
        model_path = os.path.join(run_dir, self._model_dir)

        model_load_path = os.path.join(model_path, self._model_name)
        try:
            np_data = np.load(model_load_path)
        except Exception as e:
            self.error(f"Unable to load local model from {model_load_path}: {e}")
            return None

        model = ModelLearnable()
        model[NPConstants.NUMPY_KEY] = np_data

        return model

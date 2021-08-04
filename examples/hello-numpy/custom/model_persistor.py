import os
import pickle
import numpy as np

from nvflare.apis import Model, ModelPersistor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_constant import FLConstants


class NumpyModelPersistor(ModelPersistor):
    def __init__(self, save_name=None):
        super().__init__()
        self.save_name = save_name

    def _initialize(self, fl_ctx: FLContext):
        # get save path from FLContext
        train_root = fl_ctx.get_prop(FLConstants.TRAIN_ROOT)
        log_dir = fl_ctx.get_prop(FLConstants.LOG_DIR)
        if log_dir:
            self.log_dir = os.path.join(train_root, log_dir)
        else:
            self.log_dir = train_root
        self._pkl_save_path = os.path.join(self.log_dir, self.save_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def load_model(self, fl_ctx: FLContext) -> Model:
        """
            initialize and load the Model.

        Args:
            fl_ctx: FLContext

        Returns:
            Model object
        """
        self._initialize(fl_ctx)

        model = Model()
        if os.path.exists(self._pkl_save_path):
            with open(self._pkl_save_path, "rb") as f:
                d = pickle.load(f)
                self.logger.info(f"Loading server weights: {d} \n")
            model.update(d)
        else:
            model = {"sequence": np.array([0, 1])}
        return model

    def save_model(self, model: Model, fl_ctx: FLContext):
        """
            persist the Model object

        Args:
            model: Model object
            fl_ctx: FLContext
        """
        self.logger.info(f"Saving aggregated server weights: {model} \n")
        with open(self._pkl_save_path, "wb") as f:
            pickle.dump(model, f)

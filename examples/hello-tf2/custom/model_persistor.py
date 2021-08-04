import os
import pickle
import numpy as np

import tensorflow as tf
from net import Net
from nvflare.apis.fl_context import FLContext
from nvflare.apis import Model, ModelPersistor
from nvflare.apis.fl_constant import FLConstants


class TF2ModelPersistor(ModelPersistor):
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
            self.logger.info(f"Loading server weights")
            with open(self._pkl_save_path, "rb") as f:
                var_dict = pickle.load(f)
        else:
            self.logger.info(f"Initializing server model")
            network = Net()
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            network.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
            _ = network(tf.keras.Input(shape=(28, 28)))
            var_dict = {
                network.get_layer(index=key).name: value
                for key, value in enumerate(network.get_weights())
            }
        model.update(var_dict)
        return model

    def save_model(self, model: Model, fl_ctx: FLContext):
        """
            persist the Model object

        Args:
            model: Model object
            fl_ctx: FLContext
        """
        self.logger.info(f"Saving aggregated server weights: \n {model}")
        with open(self._pkl_save_path, "wb") as f:
            pickle.dump(model, f)

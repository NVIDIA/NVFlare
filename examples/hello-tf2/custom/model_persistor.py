import os
import pickle

import tensorflow as tf
from lenet import LeNet
from nvflare.apis.fl_context import FLContext
from nvflare.apis.learnable import Learnable
from nvflare.apis.model_persistor import ModelPersistor


class TF2ModelPersistor(ModelPersistor):
    def __init__(self, save_path=None):
        super().__init__()
        self.model_path = os.path.abspath(save_path)

    def load_model(self, fl_ctx: FLContext) -> Learnable:
        """Convert initialised model into protobuf message.
        This function sets self.model to a ModelData protobuf message.

        Args:
            fl_ctx (FLContext): FL Context delivered by workflow

        Returns:
            Model: a model populated with storage pointed by fl_ctx
        """
        learnable = Learnable()
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                var_dict = pickle.load(f)
        else:
            # init network
            network = LeNet()
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            network.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
            _ = network(tf.keras.Input(shape=(28, 28)))

            var_dict = {
                str(key): value for key, value in enumerate(network.get_weights())
            }
        learnable.update(var_dict)
        return learnable

    def save_model(self, model: Learnable, fl_ctx: FLContext):
        """
            persist the Learnable object

        Args:
            model: Model object
            fl_ctx: FLContext

        """
        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)

import tensorflow as tf
import numpy as np

from net import Net
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLConstants, ShareableKey, ShareableValue
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.trainer import Trainer


class SimpleTrainer(Trainer):
    def __init__(self, epochs_per_round):
        super().__init__()
        self.epochs_per_round = epochs_per_round
        self.train_images, self.train_labels = None, None
        self.test_images, self.test_labels = None, None
        self.model = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.setup(fl_ctx)

    def setup(self, fl_ctx):
        (self.train_images, self.train_labels), (
            self.test_images,
            self.test_labels,
        ) = tf.keras.datasets.mnist.load_data()
        self.train_images, self.test_images = (
            self.train_images / 255.0,
            self.test_images / 255.0,
        )

        # simulate separate datasets for each client by dividing MNIST dataset in half
        client_name = fl_ctx.get_prop(FLConstants.CLIENT_NAME)
        if client_name == "site-1":
            self.train_images = self.train_images[:len(self.train_images)//2]
            self.train_labels = self.train_labels[:len(self.train_labels)//2]
            self.test_images = self.test_images[:len(self.test_images)//2]
            self.test_labels = self.test_labels[:len(self.test_labels)//2] 
        elif client_name == "site-2":
            self.train_images = self.train_images[len(self.train_images)//2:]
            self.train_labels = self.train_labels[len(self.train_labels)//2:]
            self.test_images = self.test_images[len(self.test_images)//2:]
            self.test_labels = self.test_labels[len(self.test_labels)//2:] 
        
        model = Net()

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
        _ = model(tf.keras.Input(shape=(28, 28)))

        self.model = model

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal) -> Shareable:
        """
        This function is an extended function from the super class.
        As a supervised learning based trainer, the train function will run
        evaluate and train engines based on model weights from `shareable`.
        After finishing training, a new `Shareable` object will be submitted
        to server for aggregation.

        Args:
            shareable: the `Shareable` object acheived from server.
            fl_ctx: the `FLContext` object achieved from server.
            abort_signal: if triggered, the training will be aborted.

        Returns:
            a new `Shareable` object to be submitted to server for aggregation.
        """

        # retrieve model weights download from server's shareable
        model_weights = shareable[ShareableKey.MODEL_WEIGHTS]

        # use previous round's client weights to replace excluded layers from server
        prev_weights = {
            self.model.get_layer(index=key).name: value for key, value in enumerate(self.model.get_weights())
        }
        for key, value in model_weights.items():
            if np.all(value == 0):
                model_weights[key] = prev_weights[key]

        # update local model weights with received weights
        self.model.set_weights(list(model_weights.values()))

        # adjust LR or other training time info as needed
        # such as callback in the fit function
        self.model.fit(
            self.train_images,
            self.train_labels,
            epochs=self.epochs_per_round,
            validation_data=(self.test_images, self.test_labels),
        )

        # report updated weights in shareable
        shareable = Shareable()
        shareable[ShareableKey.TYPE] = ShareableValue.TYPE_WEIGHTS
        shareable[ShareableKey.DATA_TYPE] = ShareableValue.DATA_TYPE_UNENCRYPTED
        shareable[ShareableKey.MODEL_WEIGHTS] = {
            self.model.get_layer(index=key).name: value for key, value in enumerate(self.model.get_weights())
        }
        self.logger.info(f"Sending shareable to server: \n {shareable[ShareableKey.MODEL_WEIGHTS]}")
        return shareable

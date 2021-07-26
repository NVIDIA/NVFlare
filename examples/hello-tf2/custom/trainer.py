import tensorflow as tf
from lenet import LeNet
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ShareableKey, ShareableValue
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
            self.setup()

    def setup(self):
        (self.train_images, self.train_labels), (
            self.test_images,
            self.test_labels,
        ) = tf.keras.datasets.mnist.load_data()
        self.train_images, self.test_images = (
            self.train_images / 255.0,
            self.test_images / 255.0,
        )

        model = LeNet()

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

        # continue on training with new weights from server via shareable
        model_weights = shareable[ShareableKey.MODEL_WEIGHTS]

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
            str(key): value for key, value in enumerate(self.model.get_weights())
        }
        return shareable

import os
import pickle
import numpy as np

from nvflare.apis.model_persistor import ModelPersistor, Learnable, FLContext

class PickleModelPersistor(ModelPersistor):
    def __init__(self, save_path):
        super().__init__()
        self.model_path = os.path.abspath(save_path)

    def load_model(self, fl_ctx: FLContext) -> Learnable:
        """
            initialize and load the Learnable.

        Args:
            fl_ctx: FLContext

        Returns:
            Model object

        """
        model = Learnable()
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                d = pickle.load(f)
                print("Loading server weights: ", d, "\n")
            model.update(d)
        else:
            model = {"sequence": np.array([0, 1])}
        return model

    def save_model(self, model: Learnable, fl_ctx: FLContext):
        """
            persist the Learnable object

        Args:
            model: Model object
            fl_ctx: FLContext

        """
        print("Saving aggregated server weights: ", model, "\n")
        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)

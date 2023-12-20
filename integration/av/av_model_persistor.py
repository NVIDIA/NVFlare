import os.path
from typing import Any

from .av_model import AVModel
from .simple_model_persistor import SimpleModelPersistor


class AVModelPersistor(SimpleModelPersistor):
    def __init__(self, file_path: str, output_path: str):
        SimpleModelPersistor.__init__(self)
        self.file_path = file_path
        if not os.path.isfile(file_path):
            raise ValueError(f"model file {file_path} does not exist")
        self.output_path = output_path

    def read_model(self) -> Any:
        print(f"loading model from {self.file_path}")
        return AVModel.load(self.file_path)

    def write_model(self, model_obj: Any):
        assert isinstance(model_obj, AVModel)
        model_obj.save(self.output_path)
        print(f"saved model in {self.output_path}")

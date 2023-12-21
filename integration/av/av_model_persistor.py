import os.path
from typing import Any

from nvflare.app_common.app_defined.model_persistor import AppDefinedModelPersistor

from .av_model import AVModel


class AVModelPersistor(AppDefinedModelPersistor):
    def __init__(self, file_path: str, output_path: str):
        AppDefinedModelPersistor.__init__(self)
        self.file_path = file_path
        if not os.path.isfile(file_path):
            raise ValueError(f"model file {file_path} does not exist")
        self.output_path = output_path

    def read_model(self) -> Any:
        self.info(f"loading model from {self.file_path}")
        return AVModel.load(self.file_path)

    def write_model(self, model_obj: Any):
        assert isinstance(model_obj, AVModel)
        model_obj.save(self.output_path)
        self.info(f"saved model in {self.output_path}")

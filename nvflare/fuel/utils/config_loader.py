from abc import ABC

from nvflare.fuel.utils.config import Config


class ConfigLoader(ABC):

    def load_config(self, file_path: str) -> Config:
        raise NotImplementedError

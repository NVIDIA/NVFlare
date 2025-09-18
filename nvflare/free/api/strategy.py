from abc import ABC, abstractmethod
from typing import List

from .proxy import Proxy


class Strategy(ABC):

    @abstractmethod
    def run(self, clients: List[Proxy], **kwargs):
        pass

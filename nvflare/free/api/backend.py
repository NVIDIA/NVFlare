from abc import abstractmethod, ABC

from .resp import Resp


class Backend(ABC):

    @abstractmethod
    def call_target(self, target_name: str, func_name: str, *args, **kwargs):
        pass

    @abstractmethod
    def call_target_with_resp(self, resp: Resp, target_name: str, func_name: str, *args, **kwargs):
        pass

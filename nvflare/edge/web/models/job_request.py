from nvflare.edge.web.models.base_model import BaseModel
from nvflare.edge.web.models.capabilities import Capabilities
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.user_info import UserInfo


class JobRequest(BaseModel):
    def __init__(
        self,
        device_info: DeviceInfo,
        user_info: UserInfo,
        capabilities: Capabilities = None,
        **kwargs,
    ):
        super().__init__()
        self.device_info = device_info
        self.user_info = user_info
        self.capabilities = capabilities

        if kwargs:
            self.update(kwargs)

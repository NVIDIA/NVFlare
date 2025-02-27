from nvflare.edge.web.models.base_model import BaseModel
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.user_info import UserInfo


class TaskRequest(BaseModel):
    def __init__(
        self,
        device_info: DeviceInfo,
        user_info: UserInfo,
        job_id: str,
        **kwargs
    ):
        super().__init__()
        self.device_info = device_info
        self.user_info = user_info
        self.job_id = job_id

        if kwargs:
            self.update(kwargs)

from nvflare.edge.web.models.base_model import BaseModel
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.user_info import UserInfo


class ResultReport(BaseModel):
    def __init__(
        self,
        device_info: DeviceInfo,
        user_info: UserInfo,
        job_id: str,
        task_id: str,
        task_name: str = None,
        result: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.device_info = device_info
        self.user_info = user_info
        self.job_id = job_id
        self.task_id = task_id
        self.task_name = task_name
        self.result = result

        if kwargs:
            self.update(kwargs)

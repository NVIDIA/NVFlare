from dataclasses import dataclass

from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.user_info import UserInfo


@dataclass
class StudyRequest:
    user_info: UserInfo
    device_info: DeviceInfo
    capabilities: dict = None
    device_state: dict = None

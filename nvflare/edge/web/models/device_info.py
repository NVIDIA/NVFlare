from nvflare.edge.web.models.base_model import DictModel


class DeviceInfo(DictModel):
    """Device information"""

    def __init__(self, device_id: str, app_name: str = None, app_version: str = None, platform: str = None):
        self.device_id = device_id
        self.app_name = app_name
        self.app_version = app_version
        self.platform = platform


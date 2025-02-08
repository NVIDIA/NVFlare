from nvflare.edge.web.models.base_model import BaseModel


class DeviceInfo(BaseModel):
    """Device information"""

    def __init__(self, device_id: str, app_name: str = None, app_version: str = None,
                 platform: str = None, platform_version: str = None, **kwargs):
        super().__init__()
        self.device_id = device_id
        self.app_name = app_name
        self.app_version = app_version
        self.platform = platform
        self.platform_version = platform_version

        if kwargs:
            self.update(kwargs)

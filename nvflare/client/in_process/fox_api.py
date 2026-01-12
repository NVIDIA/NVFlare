from typing import Any, Dict, Optional

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.client.api_spec import APISpec


class FoxClientAPI(APISpec):

    def init(self, rank: Optional[str] = None):
        pass

    def receive(self, timeout: Optional[float] = None) -> Optional[FLModel]:
        pass

    def send(self, model: FLModel, clear_cache: bool = True) -> None:
        pass

    def system_info(self) -> Dict:
        pass

    def get_config(self) -> Dict:
        pass

    def get_job_id(self) -> str:
        pass

    def get_site_name(self) -> str:
        pass

    def get_task_name(self) -> str:
        pass

    def is_running(self) -> bool:
        pass

    def is_train(self) -> bool:
        pass

    def is_evaluate(self) -> bool:
        pass

    def is_submit_model(self) -> bool:
        pass

    def log(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        pass

    def clear(self):
        pass

    def shutdown(self):
        pass

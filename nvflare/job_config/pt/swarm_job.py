from typing import List, Optional

from nvflare.app_common.ccwf.ccwf_job import CCWFJob
from nvflare.job_config.pt.model import Wrap


class SwarmJob(CCWFJob):
    def __init__(
        self,
        initial_model,
        name,
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
    ):
        super().__init__(name, min_clients, mandatory_clients)
        self.comp_ids = self.to_server(Wrap(initial_model))

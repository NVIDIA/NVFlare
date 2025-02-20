from typing import Optional

from nvflare.edge.web.models.base_model import BaseModel


class JobResponse(BaseModel):

    def __init__(
        self,
        status: str,
        job_id: str = None,
        job_name: str = None,
        method: str = None,
        job_data: Optional[dict] = None,
        retry_wait: int = None,
        **kwargs,
    ):
        super().__init__()
        self.status = status
        self.job_id = job_id
        self.job_name = job_name
        self.method = method
        self.job_data = job_data
        self.retry_wait = retry_wait

        if kwargs:
            self.update(kwargs)

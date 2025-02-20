from nvflare.edge.web.models.base_model import BaseModel


class TaskRequest(BaseModel):
    def __init__(self, session_id: str, job_id: str, **kwargs):
        super().__init__()
        self.job_id = job_id

        if kwargs:
            self.update(kwargs)

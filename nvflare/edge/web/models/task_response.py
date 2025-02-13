from nvflare.edge.web.models.base_model import BaseModel


class TaskResponse(BaseModel):

    def __init__(
        self,
        status: str,
        session_id: str = None,
        retry_wait: int = None,
        task_id: str = None,
        task_name: str = None,
        task_data: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.status = status
        self.session_id = session_id
        self.retry_wait = retry_wait
        self.task_id = task_id
        self.task_name = task_name
        self.task_data = task_data

        if kwargs:
            self.update(kwargs)

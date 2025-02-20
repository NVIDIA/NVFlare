from nvflare.edge.web.models.base_model import BaseModel


class ResultResponse(BaseModel):
    def __init__(
        self,
        status: str,
        message: str = None,
        task_id: str = None,
        task_name: str = None,
        **kwargs,
    ):
        super().__init__()
        self.status = status
        self.message = message
        self.task_id = task_id
        self.task_name = task_name

        if kwargs:
            self.update(kwargs)

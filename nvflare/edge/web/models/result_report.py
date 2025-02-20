from nvflare.edge.web.models.base_model import BaseModel


class ResultReport(BaseModel):
    def __init__(
        self,
        task_id: str,
        task_name: str = None,
        result: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.task_id = task_id
        self.task_name = task_name
        self.result = result

        if kwargs:
            self.update(kwargs)

from nvflare.edge.web.models.base_model import BaseModel


class JobRequest(BaseModel):
    def __init__(
        self,
        capabilities: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.capabilities = capabilities

        if kwargs:
            self.update(kwargs)

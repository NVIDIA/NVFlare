from nvflare.edge.web.models.base_model import BaseModel


class ErrorResponse(BaseModel):
    def __init__(self, status: str, message: str = None, details: dict = None, **kwargs):
        super().__init__()
        self.status = status
        self.message = message
        self.details = details

        if kwargs:
            self.update(kwargs)

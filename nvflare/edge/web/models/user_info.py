from nvflare.edge.web.models.base_model import BaseModel


class UserInfo(BaseModel):

    def __init__(
        self,
        user_id: str = None,
        user_name: str = None,
        access_token: str = None,
        auth_token: str = None,
        auth_session: str = None,
        **kwargs,
    ):
        super().__init__()
        self.user_id = user_id
        self.user_name = user_name
        self.access_token = access_token
        self.auth_token = auth_token
        self.auth_session = auth_session

        if kwargs:
            self.update(kwargs)

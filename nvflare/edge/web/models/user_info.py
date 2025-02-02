from nvflare.edge.web.models.base_model import DictModel


class UserInfo(DictModel):

    def __init__(self, user_id: str, user_name: str = None, auth_token: str = None, auth_session: str = None):
        self.user_id = user_id
        self.user_name = user_name
        self.auth_token = auth_token
        self.auth_session = auth_session
